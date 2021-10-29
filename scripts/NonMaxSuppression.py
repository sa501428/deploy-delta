import gc
import numpy as np


def generate_new_pick(initial_stripe, overlapping_stripes):
    new_stripe = np.squeeze(initial_stripe) + 0
    new_stripe[2] = min(new_stripe[2], np.min(overlapping_stripes[:, 2]))
    new_stripe[3] = max(new_stripe[3], np.max(overlapping_stripes[:, 3]))
    return new_stripe


def get_overlapping_indices(pick, indexes, last, boxes):
    xx1 = np.maximum(pick[0], boxes[indexes[:last], 0])
    xx2 = np.minimum(pick[1], boxes[indexes[:last], 1])
    yy1 = np.maximum(pick[2], boxes[indexes[:last], 2])
    yy2 = np.minimum(pick[3], boxes[indexes[:last], 3])
    area = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    indices = np.where(area > 0)[0]
    return indices


def update_stripe_merger(pick, indexes, last, boxes):
    overlapping_indices = get_overlapping_indices(pick, indexes, last, boxes)
    if len(overlapping_indices) > 0:
        return generate_new_pick(pick, boxes[indexes[overlapping_indices]]), overlapping_indices
    return pick, overlapping_indices


def nms(boxes, assimilate: bool = False):
    if len(boxes) == 0:
        return []
    picks = []
    indexes = np.argsort(boxes[:, 4])
    num_boxes_left = len(indexes)
    while num_boxes_left > 0:
        last = num_boxes_left - 1
        i = indexes[last]
        pick = np.squeeze(boxes[i].copy())
        num_overlap = 0
        if assimilate:
            while True:
                pick, overlapping_indices = update_stripe_merger(pick, indexes, last, boxes)
                if num_overlap == len(overlapping_indices):
                    break
                num_overlap = len(overlapping_indices)
        else:
            overlapping_indices = get_overlapping_indices(boxes[i], indexes, last, boxes)
        overlapping_indices = np.concatenate(([last], overlapping_indices))
        indexes = np.delete(indexes, overlapping_indices)
        picks.append(pick)
        num_boxes_left = len(indexes)
    return np.asarray(picks)


class Handler:
    def __init__(self):
        self.findings = {}

    def do_nms_and_print_to_file(self, outfile_name, is_domain=False, is_stripe=False):
        for key in self.findings:
            temp = list(self.findings[key])
            temp = np.vstack(temp)
            if is_stripe:
                self.findings[key] = nms(temp, assimilate=True)
            else:
                self.findings[key] = nms(temp)
        outfile_handler = open(outfile_name, 'w')
        outfile_handler.write("#chr1\tx1\tx2\tchr2\ty1\ty2\tname1\tscore\tstrand1\tstrand2\tcolor\n")
        for key in self.findings:
            self.write_to_file(outfile_handler, key[0], key[1], self.findings[key], is_domain=is_domain)
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

    def write_to_file(self, outfile_handler, chrom1, chrom2, results, is_domain=False):
        for k in range(np.shape(results)[0]):
            x1, x2, y1, y2, score = self.parse_bedpe_line(results, k)
            if is_domain:
                self.write_line(outfile_handler, chrom1, x1, y2, chrom1, x1, y2, '0,0,0', score)
            else:
                self.write_line(outfile_handler, chrom1, x1, x2, chrom2, y1, y2, '0,0,0', score)

    @staticmethod
    def write_line(out_file_handler, chrom1, x1, x2, chrom2, y1, y2, color, prediction):
        out_file_handler.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t.\t{6}\t.\t.\t{7}\n".
                               format(chrom1, x1, x2, chrom2, y1, y2, prediction, color))


class MultiResHandler(Handler):
    def __init__(self, files, output_name, radii, is_domain=False, is_stripe=False):
        super().__init__()
        for f in range(len(files)):
            self.fill_out_all_indices(files[f], len(files) - f, radii[f], is_stripe=is_stripe)
        self.do_nms_and_print_to_file(output_name, is_domain=is_domain, is_stripe=is_stripe)

    def fill_out_all_indices(self, infile, priority, offset, is_stripe):
        with open(infile) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                line_split = line.split()
                chrom1, x1, x2 = line_split[0], int(line_split[1]) - offset, int(line_split[2]) + offset
                chrom2, y1, y2 = line_split[3], int(line_split[4]) - offset, int(line_split[5]) + offset
                prediction = float(line_split[7]) + priority
                if is_stripe and x2 - x1 > y2 - y1:
                    self.add_prediction(chrom1, chrom2, y1, y2, x1, x2, prediction, offset, priority)
                else:
                    self.add_prediction(chrom1, chrom2, x1, x2, y1, y2, prediction, offset, priority)
        gc.collect()

    def write_to_file(self, outfile_handler, chrom1, chrom2, results, is_domain=False):
        for k in range(np.shape(results)[0]):
            x1, x2, y1, y2, score = self.parse_bedpe_line(results, k)
            offset, priority = int(results[k, 5]), int(results[k, 6])
            if is_domain:
                self.write_line(outfile_handler, chrom1, x1 + offset, y2 - offset, chrom1, x1 + offset, y2 - offset,
                                '0,0,0', score - priority)
            else:
                self.write_line(outfile_handler, chrom1, x1 + offset, x2 - offset, chrom2, y1 + offset, y2 - offset,
                                '0,0,0', score - priority)
