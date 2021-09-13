import concurrent
import gc
import threading
import time

import numpy as np
import strawC
from keras import backend as K
from scipy import sparse
from scipy.ndimage.measurements import label

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

    def getAmountAndClear(self, amount):
        self.__lock.acquire()
        if len(self) > amount:
            temp_list = self[:amount]
            del self[:amount]
        else:
            temp_list = self[:len(self)]
            self.clear()
        self.__lock.release()
        return temp_list

    def getAllAndClear(self):
        self.__lock.acquire()
        temp_list = self[:len(self)]
        self.clear()
        self.__lock.release()
        return temp_list


def wbce(y_true, y_pred, weight1=400, weight0=1):
    y_true = K.clip(y_true, K.epsilon(), 1 - K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    log_loss = -(y_true * K.log(y_pred) * weight1 + (1 - y_true) * K.log(1 - y_pred) * weight0)
    return K.mean(log_loss, axis=-1)


def tanh_robust_zscore(matrix, scale=0.1):
    if np.sum(matrix) < 1e-9:
        return matrix
    flattenedData = matrix.flatten()
    flattenedData = flattenedData[flattenedData > 0]
    medianVal = np.median(flattenedData)
    madVal = np.median(np.abs(flattenedData - medianVal))
    if madVal < 1e-9:
        madVal = 1
    return np.tanh(scale * (matrix - medianVal) / madVal)


class AggregatedMatrix:
    def __init__(self, shape, useArithmeticMean: bool = False):
        self.__useArithmeticMean = useArithmeticMean
        self.__num_aggregations = 0
        if self.__useArithmeticMean:
            self.__matrix = np.zeros(shape)
        else:
            self.__matrix = np.ones(shape)

    def aggregate(self, matrix):
        loop_domains = matrix[:, :, :, 0] * matrix[:, :, :, 1]
        if self.__useArithmeticMean:
            self.__matrix[:, :, :, :-1] = self.__matrix[:, :, :, :-1] + matrix
            self.__matrix[:, :, :, -1] = self.__matrix[:, :, :, -1] + loop_domains
        else:
            self.__matrix[:, :, :, :-1] = np.multiply(self.__matrix[:, :, :, :-1], matrix)
            self.__matrix[:, :, :, -1] = np.multiply(self.__matrix[:, :, :, -1], loop_domains)
        self.__num_aggregations += 1

    def getFinalResult(self):
        if self.__useArithmeticMean:
            return self.__matrix / self.__num_aggregations
        else:
            return np.power(self.__matrix, 1.0 / self.__num_aggregations)


class DeployBident:
    def __init__(self, models: list, batchSize: int, numStrawWorkers: int, filepath: str,
                 resolution: int, maxExamplesInRAM: int, matrixWidth: int, threshold: float,
                 out_files: list, preprocessMethod=tanh_robust_zscore,
                 useArithmeticMean: bool = False, norm: str = "KR",
                 numOutputChannels: int = 3):
        self.__straw_data_list = LockedList()
        self.__coords_list = LockedList()
        self.__prediction_list = LockedList()
        self.__straw_worker_done_counter = LockedList()
        self.__production_done = threading.Event()
        self.__prediction_done = threading.Event()
        self.__lock = threading.Lock()
        self.__models = models
        self.__use_arithmetic_mean = useArithmeticMean
        self.__batch_size = batchSize
        self.__num_straw_workers = numStrawWorkers
        self.__num_total_workers = numStrawWorkers + 3
        self.__hic_file = filepath
        self.__resolution = resolution
        self.__limit = maxExamplesInRAM
        self.__width = matrixWidth
        self.__threshold = threshold
        self.__norm = norm
        self.__out_files = out_files
        self.__num_output_channels = numOutputChannels
        self.__preprocess_input = preprocessMethod
        self.__nms_handlers = []
        for k in range(numOutputChannels + 1):
            self.__nms_handlers.append(Handler(resolution))
        self.__footer = self.fillOutAllIndices(filepath, resolution, norm)
        self.__num_total_slices = self.__coords_list.len()
        self.__predictFromModel()

    def grabRegion(self, chrom1, cx1, cx2, cy1, cy2, resolution):
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

    def __predictFromModel(self):
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.__num_total_workers) as executor:
            for i in range(self.__num_straw_workers):
                results.append(executor.submit(self.getAllDataSlices))
            results.append(executor.submit(self.runModelPredictions))
            results.append(executor.submit(self.generateBedpeAnnotation))
            print('All jobs submitted', flush=True)
            results[-2].result()
            results[-1].result()
        for res in results:
            res.result()
        return results

    def handlePredictions(self, currentSize, prediction, section):
        for k in range(currentSize):
            self.__prediction_list.append((prediction[k, :, :, :], section[k][1]))

    def runModelPredictions(self):
        num_done_counter = 0.0
        while not (self.__production_done.is_set() and self.getNumDataPoints() == 0):
            section = self.__straw_data_list.getAmountAndClear(self.__batch_size)
            currentSize = len(section)
            if currentSize == 0:
                time.sleep(2)
                continue

            raw_hic_input = np.zeros((currentSize, self.getWidth(), self.getWidth(), 1))
            for k in range(currentSize):
                raw_hic_input[k, :, :, 0] = section[k][0]
            agg_matrix = AggregatedMatrix((currentSize, self.getWidth(), self.getWidth(), self.__num_output_channels + 1),
                                          self.__use_arithmetic_mean)
            for model in self.__models:
                agg_matrix.aggregate(model.predict(raw_hic_input))
            result = agg_matrix.getFinalResult()

            self.handlePredictions(currentSize, result, section)
            num_done_counter += currentSize
            print('Progress ', 100. * (num_done_counter / self.__num_total_slices), '%  done', flush=True)
        time.sleep(2)
        self.__prediction_done.set()

    def getDataFromCoordinates(self, coordinates, resolution):
        (chrom1, x1, y1) = coordinates
        cx1 = x1 * resolution
        cx2 = (x1 + self.getWidth() - 1) * resolution
        cy1 = y1 * resolution
        cy2 = (y1 + self.getWidth() - 1) * resolution
        return self.grabRegion(chrom1, cx1, cx2, cy1, cy2, resolution)

    def fillOutAllIndices(self, hicfile, resolution, norm):
        chrom_dot_sizes = strawC.getChromosomes(hicfile)
        for chromosome in chrom_dot_sizes:
            chrom = chromosome.name
            if (chrom.lower() == 'all'):
                continue
            maxBin = chromosome.length // resolution + 1
            exceedBoundariesLimit = maxBin - self.getWidth()
            buffer = 50
            nearDiagDistance = 10000000 // resolution - self.getWidth()
            temp = []
            for x1 in range(0, maxBin - self.getWidth(), self.getWidth() - buffer):
                for y1 in range(x1, min(x1 + nearDiagDistance, exceedBoundariesLimit), self.getWidth() - buffer):
                    temp.append((chrom, x1, y1))
            temp.append((chrom, exceedBoundariesLimit, exceedBoundariesLimit))
            self.populateCoordinates(temp)
            del temp
        print('Near Diag Vals populated', flush=True)
        gc.collect()

        footer = {}
        for chromosome in chrom_dot_sizes:
            chrom = chromosome.name
            if (chrom.lower() == 'all'):
                continue
            print('Getting norm vector for', chrom, flush=True)
            footer[chrom] = strawC.getNormExpVectors(hicfile, chrom, chrom, "observed", norm, "BP", resolution)
        return footer


    def generateBedpeAnnotation(self):
        skip_counter = 0
        while not (self.__prediction_done.is_set() and self.getNumPredictions() == 0):
            section = self.__prediction_list.getAllAndClear()
            currentSize = len(section)

            if currentSize == 0:
                skip_counter += 1
                time.sleep(5)
                continue

            for k in range(currentSize):
                self.writePredictionToNMSHandler(section[k])

        print('Starting NMS', flush=True)
        for k in range(self.__num_output_channels + 1):
            self.__nms_handlers[k].doNMSAndPrintToFile(self.__out_files[k])

    def getAllDataSlices(self):
        while self.getNumCoordinates() > 0:
            coordinates = self.__coords_list.getAmountAndClear(1)[0]
            if coordinates is None:
                continue
            while self.getNumDataPoints() > self.getLimit():
                time.sleep(2)
            matrix = self.getDataFromCoordinates(coordinates, self.__resolution)
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
    def findMaxInConnectedComponents(prediction, chrom1, x0, y0, handler):
        labeled, ncomponents = label(prediction, np.ones((3, 3), dtype=np.int))
        for num in range(ncomponents):
            highlightedRegion = (labeled == (num + 1)) * prediction
            maxPrediction = np.max(highlightedRegion)
            maxIndices = np.where(highlightedRegion == maxPrediction)
            maxIndices = [(x, y) for x, y in zip(maxIndices[0], maxIndices[1])]
            for maxIndex in maxIndices:
                rowStart = x0 + maxIndex[0]
                rowEnd = rowStart + 1
                columnStart = y0 + maxIndex[1]
                columnEnd = columnStart + 1
                if rowStart < columnStart:
                    handler.addPrediction(chrom1, chrom1, rowStart, rowEnd, columnStart, columnEnd, maxPrediction)
                else:
                    handler.addPrediction(chrom1, chrom1, columnStart, columnEnd, rowStart, rowEnd, maxPrediction)

    @staticmethod
    def findBoundInConnectedComponents(prediction, chrom1, x0, y0, handler):
        labeled, ncomponents = label(prediction, np.ones((3, 3), dtype=np.int))
        for num in range(ncomponents):
            highlightedRegion = (labeled == (num + 1)) * prediction
            maxPrediction = np.max(highlightedRegion)
            indices = np.where(highlightedRegion > 0)
            rowStart = int(x0 + np.min(indices[0]))
            rowEnd = max(int(x0 + np.max(indices[0])), rowStart + 1)
            colStart = int(y0 + np.min(indices[1]))
            colEnd = max(int(y0 + np.max(indices[1])), colStart + 1)
            handler.addPrediction(chrom1, chrom1, rowStart, rowEnd, colStart, colEnd, maxPrediction)

    def writePredictionToNMSHandler(self, section):
        prediction = section[0]
        prediction[prediction < self.getThreshold()] = 0
        chrom1 = section[1][0]
        x1 = int(section[1][1])
        y1 = int(section[1][2])
        for k in range(self.__num_output_channels + 1):
            if k == 2 and self.__num_output_channels == 3:
                self.findBoundInConnectedComponents(prediction[:, :, k], chrom1, x1, y1, self.__nms_handlers[k])
            else:
                self.findMaxInConnectedComponents(prediction[:, :, k], chrom1, x1, y1, self.__nms_handlers[k])

    def getThreshold(self):
        return self.__threshold

    def getWidth(self):
        return self.__width

    def getLimit(self):
        return self.__limit

    def getNumCoordinates(self):
        return self.__coords_list.len()

    def getNumDataPoints(self):
        return self.__straw_data_list.len()

    def getNumPredictions(self):
        return self.__prediction_list.len()

    def populateCoordinates(self, coords):
        self.__coords_list.extend(coords)