import gc

import numpy as np


class FastNMS:
    def __init__(self, resolution, threshold):
        self.__resolution = resolution
        self.__threshold = threshold

    def run(self, boxes):
        if len(boxes) == 0:
            return []
        boxes[:, :4] = boxes[:, :4] / self.__resolution
        pick = []
        x1 = boxes[:, 0]
        x2 = boxes[:, 1]
        y1 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = (x2 - x1) * (y2 - y1) + 1
        indexes = np.argsort(boxes[:, 4])
        while len(indexes) > 0:
            last = len(indexes) - 1
            i = indexes[last]
            pick.append(i)
            xx1 = np.maximum(x1[i], x1[indexes[:last]])
            yy1 = np.maximum(y1[i], y1[indexes[:last]])
            xx2 = np.minimum(x2[i], x2[indexes[:last]])
            yy2 = np.minimum(y2[i], y2[indexes[:last]])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            overlap = (w * h) / area[indexes[:last]]
            indexes = np.delete(indexes, np.concatenate(([last], np.where(overlap > self.__threshold)[0])))
        boxes[:, :4] = boxes[:, :4] * self.__resolution
        return boxes[pick]


class Handler:
    def __init__(self, resolution, threshold=.3):
        self.findings = {}
        self.resolution = resolution
        self.nms = FastNMS(1, threshold)

    def do_nms_and_print_to_file(self, outfile_name):
        for key in self.findings:
            temp = list(self.findings[key])
            temp = np.vstack(temp)
            self.findings[key] = self.nms.run(temp)

        outfile_handler = open(outfile_name, 'w')
        outfile_handler.write("#chr1\tx1\tx2\tchr2\ty1\ty2\tname1\tscore\tstrand1\tstrand2\tcolor\n")
        for key in self.findings:
            self.write_to_file(outfile_handler, key[0], key[1], self.findings[key])
        outfile_handler.close()

    def add_prediction(self, chrom1, chrom2, x1, x2, y1, y2, prediction, offset=0, priority=0):
        local_key = (chrom1, chrom2)
        if local_key not in self.findings:
            self.findings[local_key] = set()
        self.findings[local_key].add((x1, x2, y1, y2, prediction, offset, priority))

    @staticmethod
    def parse_bedpe_line(data, index):
        x1, x2 = int(data[index, 0]), int(data[index, 1])
        y1, y2, = int(data[index, 2]), int(data[index, 3])
        score = float(data[index, 4])
        return x1, x2, y1, y2, score

    def write_to_file(self, outfile_handler, chrom1, chrom2, results):
        results[:, :4] = results[:, :4] * self.resolution
        for k in range(np.shape(results)[0]):
            x1, x2, y1, y2, score = self.parse_bedpe_line(results, k)
            self.write_line(outfile_handler, chrom1, x1, x2, chrom2, y1, y2, '0,0,0', score)

    @staticmethod
    def write_line(out_file_handler, chrom1, x1, x2, chrom2, y1, y2, color, prediction):
        out_file_handler.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t.\t{6}\t.\t.\t{7}\n".
                               format(chrom1, x1, x2, chrom2, y1, y2, prediction, color))


class MultiResHandler(Handler):
    def __init__(self, files, output_name, threshold=.3, radii=None):
        super().__init__(1, threshold)
        if radii is None:
            for file in files:
                self.fill_out_all_indices(file, 0, 0)
        else:
            for f in range(len(files)):
                self.fill_out_all_indices(files[f], len(files) - f, radii[f])
        self.do_nms_and_print_to_file(output_name)

    def fill_out_all_indices(self, infile, priority, offset):
        with open(infile) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                line_split = line.split()
                chrom1, x1, x2 = line_split[0], int(line_split[1]) - offset, int(line_split[2]) + offset
                chrom2, y1, y2 = line_split[3], int(line_split[4]) - offset, int(line_split[5]) + offset
                prediction = float(line_split[7]) + priority
                self.add_prediction(chrom1, chrom2, x1, x2, y1, y2, prediction, offset, priority)
        gc.collect()

    def write_to_file(self, outfile_handler, chrom1, chrom2, results):
        results[:, :4] = results[:, :4] * self.resolution
        for k in range(np.shape(results)[0]):
            x1, x2, y1, y2, score = self.parse_bedpe_line(results, k)
            offset, priority = int(results[k, 5]), int(results[k, 6])
            self.write_line(outfile_handler, chrom1, x1 + offset, x2 - offset,
                            chrom2, y1 + offset, y2 - offset, '0,0,0', score - priority)
