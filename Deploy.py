import sys
from keras import models, layers
from keras import backend as keras
import tensorflow as tf
import time
import glob
import numpy as np

from scripts.StrawMLTools import DeploySpears
from scripts.NonMaxSuppression import MultiResHandler

print('done importing')

# python3 Deploy.py <file.hic> </dir/models> </dir/working> <stem> <res,> <norm>

MATRIX_WIDTH = 500
FILEPATH = sys.argv[1]
MODEL_DIR = sys.argv[2]
WORKING_DIR = sys.argv[3]
STEM = sys.argv[4]
RESOLUTIONS = sys.argv[5].split(",")
THRESHOLD = 0.8
# WORKING_DIR = "/home/alyssa/loop_caller/machine-learning-deployment/"
# CELL_LINE = 'tridentv1_results/test'#GM12878_intact_30' #'gm12878_intra_nofrag_30'
# FILEPATH = '/home/alyssa/loop_caller/machine-learning-deployment/intact_test.hic'
# '/mnt/HIC2000/HiSeq/GM12878_2020_combined_maps/GM12878_quadRE_SDSallconc_10.5.20/inter_30.hic'
# '/mnt/HIC2000/HiSeq/GM12878_2020_combined_maps/GM12878_intact_combined_18.7B_10.5.20/inter_30.hic'
# '/mnt/HIC2000/HiSeq/GM12878_2020_combined_maps/GM12878_MNase_0.1_10.5.20/inter_30.hic'
# '/mnt/HIC2000/HiSeq/GM12878_2020_combined_maps/GM12878_quadRE_noSDS_10.5.20/inter_30.hic'
# '/mnt/HIC2000/HiSeq/GM12878_2020_combined_maps/GM12878_intact_combined_18.7B_10.5.20/inter_30.hic'
# '/mnt/HIC0000/HiSeq/MboI_magic_biorep/inter_30.hic' #'/mnt/HIC0000/mega/IMR90/DRGN/inter_30.hic'
NORMALIZATION_TYPE = sys.argv[6]  # "KR"
MAX_EXAMPLES_IN_RAM = 30
BATCH_SIZE = 30
NUM_STRAW_WORKERS = 10


class WBCE:
    def __init__(self, weight: int = 120):
        self.weight = weight

    def func(self, y_true, y_pred):
        y_true = keras.clip(y_true, keras.epsilon(), 1 - keras.epsilon())
        y_pred = keras.clip(y_pred, keras.epsilon(), 1 - keras.epsilon())
        return -keras.mean(y_true[:, :, :, 0] * keras.log(y_pred[:, :, :, 0]) * self.weight +
                           (1 - y_true[:, :, :, 0]) * keras.log(1 - y_pred[:, :, :, 0]))


def preprocessing_method(matrix):  # , scale=0.1
    if np.sum(matrix) < 1e-9:
        return matrix
    flattened_data = matrix.flatten()
    flattened_data = flattened_data[flattened_data > 0]
    median_val = np.median(flattened_data)
    mad_val = np.median(np.abs(flattened_data - median_val))
    if mad_val < 1e-9:
        mad_val = 1
    # return np.tanh(scale * (matrix - median_val) / mad_val)
    return (matrix - median_val) / mad_val
    # TODO - try global values in the future


def load_models():
    model_list = []
    for file in glob.glob(MODEL_DIR + "/*.h5"):
        print("Using Model", file, flush=True)
        model_list.append(models.load_model(file,
                                            custom_objects={'wbce': WBCE().func,
                                                            'leaky_relu': tf.nn.leaky_relu,
                                                            'LeakyReLU': layers.LeakyReLU()}))
    return model_list


def deploy_at_res(models, res):
    outpath = WORKING_DIR + "/" + STEM
    loop_file = outpath + '_loops_' + str(res) + '.bedpe'
    domain_file = outpath + '_domains_' + str(res) + '.bedpe'
    stripe_file = outpath + '_stripes_' + str(res) + '.bedpe'
    loop_domain_file = outpath + '_loop_domains_' + str(res) + '.bedpe'
    print('got to deployment')
    print("Doing resolution", res, "for", STEM, flush=True)
    DeploySpears(models=models, batchSize=BATCH_SIZE,
                 numStrawWorkers=NUM_STRAW_WORKERS, filepath=FILEPATH,
                 resolution=res,
                 maxExamplesInRAM=MAX_EXAMPLES_IN_RAM, matrixWidth=MATRIX_WIDTH,
                 threshold=THRESHOLD,
                 out_files=[loop_file, domain_file, stripe_file, loop_domain_file],
                 preprocessMethod=preprocessing_method,
                 useArithmeticMean=False,
                 norm=NORMALIZATION_TYPE,
                 numOutputChannels=3)
    return loop_file, domain_file, stripe_file, loop_domain_file


def merge_lists(loop_lists, domain_lists, stripe_lists, loop_domain_lists, loop_radii, domain_radii):
    if len(loop_lists) > 1:
        outpath = WORKING_DIR + "/" + STEM
        MultiResHandler(loop_lists, outpath + '_loops_merged.bedpe', threshold=1e-10,
                        radii=loop_radii)
        MultiResHandler(domain_lists, outpath + '_domains_merged.bedpe', threshold=1e-10,
                        radii=domain_radii)
        MultiResHandler(stripe_lists, outpath + '_stripes_merged.bedpe', threshold=1e-10,
                        radii=loop_radii)
        MultiResHandler(loop_domain_lists, outpath + '_loop_domains_merged.bedpe', threshold=1e-10,
                        radii=domain_radii)


class DeployTridentFeatures:
    def __init__(self):
        print('initialize deployment', flush=True)
        models = load_models()

        resolutions = []
        for res_string in RESOLUTIONS:
            resolutions.append(int(res_string))
        resolutions.sort()
        print("Using resolutions", resolutions, flush=True)

        loop_lists = []
        domain_lists = []
        stripe_lists = []
        loop_domains_lists = []
        loop_radii = []
        domain_radii = []
        for res in resolutions:
            start_time = time.time()
            loops, domains, stripes, loop_domains = deploy_at_res(models, res)

            runtime = time.time() - start_time
            print("Executed resolution {} in {} seconds".format(res, runtime))
            print("-----------------------------------------", flush=True)

            loop_lists.append(loops)
            domain_lists.append(domains)
            stripe_lists.append(stripes)
            loop_domains_lists.append(loop_domains)
            loop_radii.append(res)
            domain_radii.append(3 * res)
        merge_lists(loop_lists, domain_lists, stripe_lists, loop_domains_lists,
                         loop_radii, domain_radii)


print('starting deployment', flush=True)
DeployTridentFeatures()
