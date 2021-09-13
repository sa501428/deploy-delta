import gc
import threading
import time

import numpy as np
import strawC
from scipy import sparse
from scipy.ndimage.measurements import label
from concurrent.futures import ThreadPoolExecutor

from .NonMaxSuppression import Handler


class LockedList(list):
    def __init__(self):
        super().__init__()
        self.__lock = threading.Lock()

    def append(self, item):
        self.__lock.acquire()
        super().append(item)
        self.__lock.release()

    def extend(self, item):
        self.__lock.acquire()
        super().extend(item)
        self.__lock.release()

    def len(self):
        self.__lock.acquire()
        temp_length = len(self)
        self.__lock.release()
        return temp_length

    def get_amount_and_clear(self, amount):
        self.__lock.acquire()
        if len(self) > amount:
            temp_list = self[:amount]
            del self[:amount]
        else:
            temp_list = self[:len(self)]
            self.clear()
        self.__lock.release()
        return temp_list

    def get_all_and_clear(self):
        self.__lock.acquire()
        temp_list = self[:len(self)]
        self.clear()
        self.__lock.release()
        return temp_list


class AggregatedMatrix:
    def __init__(self, shape, use_arithmetic_mean: bool = False):
        self.__useArithmeticMean = use_arithmetic_mean
        self.__num_aggregations = 0
        if self.__useArithmeticMean:
            self.__matrix = np.zeros(shape)
        else:
            self.__matrix = np.ones(shape)

    def aggregate(self, matrix, index):
        if self.__useArithmeticMean:
            self.__matrix[:, :, :, index] = self.__matrix[:, :, :, index] + matrix
        else:
            self.__matrix[:, :, :, index] = np.multiply(self.__matrix[:, :, :, index], matrix)
        if index == 0:
            self.__num_aggregations += 1

    def __calc_loop_domains(self):
        self.__matrix[:, :, :, -1] = self.__matrix[:, :, :, 0] * self.__matrix[:, :, :, 1]

    def get_final_result(self):
        if self.__useArithmeticMean:
            self.__matrix = self.__matrix / self.__num_aggregations
        else:
            self.__matrix = np.power(self.__matrix, 1.0 / self.__num_aggregations)
        self.__calc_loop_domains()
        return self.__matrix


class DeploySpears:
    def __init__(self, all_model_sets: list, batch_size: int, num_straw_workers: int, filepath: str,
                 resolution: int, max_examples_in_ram: int, matrix_width: int, threshold: float,
                 out_files: list, preprocess_method,
                 use_arithmetic_mean: bool = False, norm: str = "KR",
                 num_output_channels: int = 3):
        self.__straw_data_list = LockedList()
        self.__coords_list = LockedList()
        self.__prediction_list = LockedList()
        self.__straw_worker_done_counter = LockedList()
        self.__production_done = threading.Event()
        self.__prediction_done = threading.Event()
        self.__lock = threading.Lock()
        self.__all_model_sets = all_model_sets
        self.__use_arithmetic_mean = use_arithmetic_mean
        self.__batch_size = batch_size
        self.__num_straw_workers = num_straw_workers
        self.__num_total_workers = num_straw_workers + 3
        self.__hic_file = filepath
        self.__resolution = resolution
        self.__limit = max_examples_in_ram
        self.__width = matrix_width
        self.__threshold = threshold
        self.__norm = norm
        self.__out_files = out_files
        self.__num_output_channels = num_output_channels
        self.__preprocess_input = preprocess_method
        self.__nms_handlers = []
        for k in range(num_output_channels + 1):
            self.__nms_handlers.append(Handler(resolution))
        self.__footer = self.fill_out_all_indices(filepath, resolution, norm)
        self.__num_total_slices = self.__coords_list.len()
        self.__predict_from_model()

    def grab_region(self, chrom1, cx1, cx2, cy1, cy2, resolution):
        footer = self.__footer[chrom1]
        result = strawC.getRecords(self.__hic_file, cx1, cx2, cy1, cy2,
                                   resolution, footer.foundFooter, footer.version,
                                   footer.c1, footer.c2, footer.numBins1, footer.numBins2,
                                   footer.myFilePos, footer.unit, footer.norm,
                                   footer.matrixType, footer.c1Norm, footer.c2Norm,
                                   footer.expectedValues)
        row_indices, col_indices, data = list(), list(), list()
        for record in result:
            if cx1 <= record.binX <= cx2 and cy1 <= record.binY <= cy2:
                row_indices.append(record.binX)
                col_indices.append(record.binY)
                data.append(record.counts)
            if record.binX != record.binY and cx1 <= record.binY <= cx2 and cy1 <= record.binX <= cy2:
                row_indices.append(record.binY)
                col_indices.append(record.binX)
                data.append(record.counts)

        row_indices = (np.asarray(row_indices) - cx1) / resolution
        col_indices = (np.asarray(col_indices) - cy1) / resolution
        matrix = sparse.coo_matrix((data, (row_indices.astype(int), col_indices.astype(int))),
                                   shape=(self.__width, self.__width)).toarray()
        matrix[np.isnan(matrix)] = 0
        matrix[np.isinf(matrix)] = 0
        return matrix

    def __predict_from_model(self):
        results = []
        with ThreadPoolExecutor(max_workers=self.__num_total_workers) as executor:
            for i in range(self.__num_straw_workers):
                results.append(executor.submit(self.get_all_data_slices))
            results.append(executor.submit(self.run_model_predictions))
            results.append(executor.submit(self.generate_bedpe_annotation))
            print('All jobs submitted', flush=True)
            results[-2].result()
            results[-1].result()
        for res in results:
            res.result()
        return results

    def handle_predictions(self, current_size, prediction, section):
        for k in range(current_size):
            self.__prediction_list.append((prediction[k, :, :, :], section[k][1]))

    def run_model_predictions(self):
        num_done_counter = 0.0
        while not (self.__production_done.is_set() and self.get_num_data_points() == 0):
            section = self.__straw_data_list.get_amount_and_clear(self.__batch_size)
            current_size = len(section)
            if current_size == 0:
                time.sleep(2)
                continue

            raw_hic_input = np.zeros((current_size, self.get_width(), self.get_width(), 1))
            for k in range(current_size):
                raw_hic_input[k, :, :, 0] = section[k][0]
            agg_matrix = AggregatedMatrix((current_size, self.get_width(), self.get_width(),
                                           self.__num_output_channels + 1), self.__use_arithmetic_mean)
            for p in range(len(self.__all_model_sets)):
                for model in self.__all_model_sets[p]:
                    agg_matrix.aggregate(model.predict(raw_hic_input), p)
            result = agg_matrix.get_final_result()

            self.handle_predictions(current_size, result, section)
            num_done_counter += current_size
            print('Progress ', 100. * (num_done_counter / self.__num_total_slices), '%  done', flush=True)
        time.sleep(2)
        self.__prediction_done.set()

    def get_data_from_coordinates(self, coordinates, resolution):
        (chrom1, x1, y1) = coordinates
        cx1 = x1 * resolution
        cx2 = (x1 + self.get_width() - 1) * resolution
        cy1 = y1 * resolution
        cy2 = (y1 + self.get_width() - 1) * resolution
        return self.grab_region(chrom1, cx1, cx2, cy1, cy2, resolution)

    def fill_out_all_indices(self, hic_file, resolution, norm):
        chrom_dot_sizes = strawC.getChromosomes(hic_file)
        for chromosome in chrom_dot_sizes:
            chrom = chromosome.name
            if chrom.lower() == 'all':
                continue
            max_bin = chromosome.length // resolution + 1
            exceed_boundaries_limit = max_bin - self.get_width()
            buffer = 50
            near_diag_distance = 10000000 // resolution - self.get_width()
            temp = []
            for x1 in range(0, max_bin - self.get_width(), self.get_width() - buffer):
                for y1 in range(x1, min(x1 + near_diag_distance, exceed_boundaries_limit), self.get_width() - buffer):
                    temp.append((chrom, x1, y1))
            temp.append((chrom, exceed_boundaries_limit, exceed_boundaries_limit))
            self.populate_coordinates(temp)
            del temp
        print('Near Diagonal Values populated', flush=True)
        gc.collect()

        footer = {}
        for chromosome in chrom_dot_sizes:
            chrom = chromosome.name
            if chrom.lower() == 'all':
                continue
            print('Getting norm vector for', chrom, flush=True)
            footer[chrom] = strawC.getNormExpVectors(hic_file, chrom, chrom, "observed", norm, "BP", resolution)
        return footer

    def generate_bedpe_annotation(self):
        skip_counter = 0
        while not (self.__prediction_done.is_set() and self.get_num_predictions() == 0):
            section = self.__prediction_list.get_all_and_clear()
            current_size = len(section)

            if current_size == 0:
                skip_counter += 1
                time.sleep(5)
                continue

            for k in range(current_size):
                self.write_prediction_to_nms_handler(section[k])

        print('Starting NMS', flush=True)
        for k in range(self.__num_output_channels + 1):
            self.__nms_handlers[k].doNMSAndPrintToFile(self.__out_files[k])

    def get_all_data_slices(self):
        while self.get_num_coordinates() > 0:
            coordinates = self.__coords_list.get_amount_and_clear(1)[0]
            if coordinates is None:
                continue
            while self.get_num_data_points() > self.get_limit():
                time.sleep(2)
            matrix = self.get_data_from_coordinates(coordinates, self.__resolution)
            if matrix is None or not (type(matrix) is np.ndarray):
                continue
            self.__straw_data_list.append((self.__preprocess_input(matrix.copy()), coordinates))
            del matrix
            gc.collect()
        with self.__lock:
            self.__straw_worker_done_counter.append(1)
            if self.__straw_worker_done_counter.len() == self.__num_straw_workers:
                self.__production_done.set()
                print("DONE GETTING ALL REGIONS VIA STRAW!!!", flush=True)

    @staticmethod
    def find_max_in_connected_components(prediction, chrom1, x0, y0, handler):
        labeled, n_components = label(prediction, np.ones((3, 3), dtype=np.int))
        for num in range(n_components):
            highlighted_region = (labeled == (num + 1)) * prediction
            max_prediction = np.max(highlighted_region)
            max_indices = np.where(highlighted_region == max_prediction)
            max_indices = [(x, y) for x, y in zip(max_indices[0], max_indices[1])]
            for maxIndex in max_indices:
                row_start = x0 + maxIndex[0]
                row_end = row_start + 1
                column_start = y0 + maxIndex[1]
                column_end = column_start + 1
                if row_start < column_start:
                    handler.addPrediction(chrom1, chrom1, row_start, row_end, column_start, column_end, max_prediction)
                else:
                    handler.addPrediction(chrom1, chrom1, column_start, column_end, row_start, row_end, max_prediction)

    @staticmethod
    def find_bound_in_connected_components(prediction, chrom1, x0, y0, handler):
        labeled, n_components = label(prediction, np.ones((3, 3), dtype=np.int))
        for num in range(n_components):
            highlighted_region = (labeled == (num + 1)) * prediction
            max_prediction = np.max(highlighted_region)
            indices = np.where(highlighted_region > 0)
            row_start = int(x0 + np.min(indices[0]))
            row_end = max(int(x0 + np.max(indices[0])), row_start + 1)
            col_start = int(y0 + np.min(indices[1]))
            col_end = max(int(y0 + np.max(indices[1])), col_start + 1)
            handler.addPrediction(chrom1, chrom1, row_start, row_end, col_start, col_end, max_prediction)

    def write_prediction_to_nms_handler(self, section):
        prediction = section[0]
        prediction[prediction < self.get_threshold()] = 0
        chrom1 = section[1][0]
        x1 = int(section[1][1])
        y1 = int(section[1][2])
        for k in range(self.__num_output_channels + 1):
            if k == 2 and self.__num_output_channels == 3:
                self.find_bound_in_connected_components(prediction[:, :, k], chrom1, x1, y1, self.__nms_handlers[k])
            else:
                self.find_max_in_connected_components(prediction[:, :, k], chrom1, x1, y1, self.__nms_handlers[k])

    def get_threshold(self):
        return self.__threshold

    def get_width(self):
        return self.__width

    def get_limit(self):
        return self.__limit

    def get_num_coordinates(self):
        return self.__coords_list.len()

    def get_num_data_points(self):
        return self.__straw_data_list.len()

    def get_num_predictions(self):
        return self.__prediction_list.len()

    def populate_coordinates(self, coords):
        self.__coords_list.extend(coords)
