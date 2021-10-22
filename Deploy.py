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
THRESHOLD = 0.9
NORMALIZATION_TYPE = sys.argv[6]  # "KR"
MAX_EXAMPLES_IN_RAM = 30
BATCH_SIZE = 30
NUM_STRAW_WORKERS = 10
FEATURE_TYPES = ["loops", "domains", "stripes"]


class WBCE:
    def __init__(self, weight: int = 120):
        self.weight = weight

    def func(self, y_true, y_pred):
        y_true = keras.clip(y_true, keras.epsilon(), 1 - keras.epsilon())
        y_pred = keras.clip(y_pred, keras.epsilon(), 1 - keras.epsilon())
        return -keras.mean(y_true[:, :, :, 0] * keras.log(y_pred[:, :, :, 0]) * self.weight +
                           (1 - y_true[:, :, :, 0]) * keras.log(1 - y_pred[:, :, :, 0]))


def preprocessing_method(matrix, m_width: int = 500):  # , scale=0.1
    result = np.zeros((m_width, m_width, 3))
    if np.sum(matrix) < 1e-9:
        return result
    flattened_data = matrix.flatten()
    flattened_data = flattened_data[flattened_data > 0]
    median_val = np.median(flattened_data)
    mad_val = np.median(np.abs(flattened_data - median_val))
    if mad_val < 1e-9:
        if median_val < 1e-9:
            mad_val = 1
        else:
            mad_val = median_val / 2
    # print('median', median_val, 'mad', mad_val)
    result[:, :, 0] = matrix / (5 * mad_val)
    result[:, :, 1] = matrix / (10 * mad_val)
    result[:, :, 2] = matrix / (30 * mad_val)
    return np.tanh(result - 2)


def load_models():
    all_sets_of_models = []
    wbce = WBCE()
    print("Loading all Models")
    for feature in FEATURE_TYPES:
        model_list = []
        for file in glob.glob(MODEL_DIR + "/*" + feature + "*.h5"):
            print("Using Model", file, flush=True)
            model_list.append(models.load_model(file,
                                                custom_objects={'wbce': wbce.func,
                                                                'func': wbce.func,
                                                                'leaky_relu': tf.nn.leaky_relu,
                                                                'LeakyReLU': layers.LeakyReLU()}))
        all_sets_of_models.append(model_list)
    return all_sets_of_models


def deploy_at_res(all_model_sets, res):
    out_path = WORKING_DIR + "/" + STEM
    loop_file = out_path + '_loops_' + str(res) + '.bedpe'
    domain_file = out_path + '_domains_' + str(res) + '.bedpe'
    stripe_file = out_path + '_stripes_' + str(res) + '.bedpe'
    loop_domain_file = out_path + '_loop_domains_' + str(res) + '.bedpe'
    print('got to deployment')
    print("Doing resolution", res, "for", STEM, flush=True)
    DeploySpears(all_model_sets=all_model_sets, batch_size=BATCH_SIZE,
                 num_straw_workers=NUM_STRAW_WORKERS, filepath=FILEPATH,
                 resolution=res,
                 max_examples_in_ram=MAX_EXAMPLES_IN_RAM, matrix_width=MATRIX_WIDTH,
                 threshold=THRESHOLD,
                 out_files=[loop_file, domain_file, stripe_file, loop_domain_file],
                 preprocess_method=preprocessing_method,
                 use_arithmetic_mean=False,
                 norm=NORMALIZATION_TYPE,
                 num_output_channels=3)
    return loop_file, domain_file, stripe_file, loop_domain_file


def merge_lists(loop_lists, domain_lists, stripe_lists, loop_domain_lists, loop_radii, domain_radii):
    if len(loop_lists) > 1:
        out_path = WORKING_DIR + "/" + STEM
        MultiResHandler(loop_lists, out_path + '_loops_merged.bedpe', threshold=1e-10,
                        radii=loop_radii)
        MultiResHandler(domain_lists, out_path + '_domains_merged.bedpe', threshold=1e-10,
                        radii=domain_radii, is_domain=True)
        MultiResHandler(stripe_lists, out_path + '_stripes_merged.bedpe', threshold=1e-10,
                        radii=loop_radii, is_stripe=True)
        MultiResHandler(loop_domain_lists, out_path + '_loop_domains_merged.bedpe', threshold=1e-10,
                        radii=domain_radii, is_domain=True)


class DeployTridentFeatures:
    def __init__(self):
        print('initialize deployment', flush=True)
        all_model_sets = load_models()

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
            loops, domains, stripes, loop_domains = deploy_at_res(all_model_sets, res)

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
