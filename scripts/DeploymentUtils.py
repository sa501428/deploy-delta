from keras import models, layers
from keras import backend as keras
import tensorflow as tf
import time
import glob
import numpy as np

from scripts.StrawMLTools import DeploySpears
from scripts.NonMaxSuppression import MultiResHandler


class WBCE:
    def __init__(self, weight: int = 120):
        self.weight = weight

    def func(self, y_true, y_pred):
        y_true = keras.clip(y_true, keras.epsilon(), 1 - keras.epsilon())
        y_pred = keras.clip(y_pred, keras.epsilon(), 1 - keras.epsilon())
        return -keras.mean(y_true[:, :, :, 0] * keras.log(y_pred[:, :, :, 0]) * self.weight +
                           (1 - y_true[:, :, :, 0]) * keras.log(1 - y_pred[:, :, :, 0]))


def insitu_preprocessing_method(matrix, m_width: int = 500):  # , scale=0.1
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


def intact_preprocessing_method(matrix, m_width: int = 500):  # , scale=0.1
    result = np.zeros((m_width, m_width, 3))
    if np.sum(matrix) < 1e-9:
        return result
    flattened_data = matrix.flatten()
    flattened_data = flattened_data[flattened_data > 0]
    median_val = np.median(flattened_data)
    mad_val = np.median(np.abs(flattened_data - median_val))
    if mad_val < 1:
        mad_val = 1
    m2 = matrix / mad_val
    result[:, :, 0] = np.log(1 + m2)
    result[:, :, 1] = m2 / 7
    result[:, :, 2] = m2 * m2 / 200
    return np.tanh(result - 2)


def load_models(feature_types: list, model_directory: str) -> list:
    all_models = []
    wbce = WBCE()
    print("Loading all Models")
    for feature in feature_types:
        model_list = []
        for file in glob.glob(model_directory + "/*" + feature + "*.h5"):
            print("Using Model", file, flush=True)
            model_list.append(models.load_model(file,
                                                custom_objects={'wbce': wbce.func,
                                                                'func': wbce.func,
                                                                'leaky_relu': tf.nn.leaky_relu,
                                                                'LeakyReLU': layers.LeakyReLU()}))
        all_models.append(model_list)
    return all_models


def deploy_at_res(filepath: str, all_models: list, matrix_width: int, res: int, working_directory: str,
                  normalization_type: str, stem: str, max_examples_in_ram: int, num_straw_workers: int,
                  threshold: float, batch_size: int, use_insitu_preprocessing: bool):
    out_path = working_directory + "/" + stem
    loop_file = out_path + '_loops_' + str(res) + '.bedpe'
    domain_file = out_path + '_domains_' + str(res) + '.bedpe'
    stripe_file = out_path + '_stripes_' + str(res) + '.bedpe'
    loop_domain_file = out_path + '_loop_domains_' + str(res) + '.bedpe'
    print('got to deployment')
    print("Doing resolution", res, "for", stem, flush=True)
    method = intact_preprocessing_method
    if use_insitu_preprocessing:
        method = insitu_preprocessing_method

    DeploySpears(all_model_sets=all_models, batch_size=batch_size,
                 num_straw_workers=num_straw_workers, filepath=filepath,
                 resolution=res,
                 max_examples_in_ram=max_examples_in_ram, matrix_width=matrix_width,
                 threshold=threshold,
                 out_files=[loop_file, domain_file, stripe_file, loop_domain_file],
                 preprocess_method=method,
                 use_arithmetic_mean=False,
                 norm=normalization_type,
                 num_output_channels=3)
    return loop_file, domain_file, stripe_file, loop_domain_file


def merge_lists(working_directory, stem, loop_lists, domain_lists, stripe_lists,
                loop_domain_lists, loop_radii, domain_radii):
    if len(loop_lists) > 1:
        out_path = working_directory + "/" + stem
        MultiResHandler(loop_lists, out_path + '_loops_merged.bedpe', radii=loop_radii)
        MultiResHandler(domain_lists, out_path + '_domains_merged.bedpe', radii=domain_radii, is_domain=True)
        MultiResHandler(stripe_lists, out_path + '_stripes_merged.bedpe', radii=loop_radii, is_stripe=True)
        MultiResHandler(loop_domain_lists, out_path + '_loop_domains_merged.bedpe', radii=domain_radii, is_domain=True)


class DeployTridentFeatures:
    def __init__(self, filepath: str, all_models: list, resolution_strings: list,
                 stem: str, working_directory: str, normalization_type: str, threshold: float, matrix_width: int,
                 max_examples_in_ram: int, num_straw_workers: int, batch_size: int,
                 use_insitu_preprocessing: bool):
        resolutions = []
        for res_string in resolution_strings:
            resolutions.append(int(res_string))
        resolutions.sort()

        loop_lists = []
        domain_lists = []
        stripe_lists = []
        loop_domains_lists = []
        loop_radii = []
        domain_radii = []
        for res in resolutions:
            start_time = time.time()
            loops, domains, stripes, loop_domains = deploy_at_res(filepath, all_models, matrix_width, res,
                                                                  working_directory, normalization_type, stem,
                                                                  max_examples_in_ram, num_straw_workers,
                                                                  threshold, batch_size, use_insitu_preprocessing)

            runtime = time.time() - start_time
            print("Executed resolution {} in {} seconds".format(res, runtime))
            print("-----------------------------------------", flush=True)

            loop_lists.append(loops)
            domain_lists.append(domains)
            stripe_lists.append(stripes)
            loop_domains_lists.append(loop_domains)
            loop_radii.append(res)
            domain_radii.append(3 * res)
        merge_lists(working_directory, stem, loop_lists, domain_lists, stripe_lists, loop_domains_lists,
                    loop_radii, domain_radii)
