import sys
from scripts.DeploymentUtils import DeployTridentFeatures, load_models

import tensorflow as tf
import os
from tensorflow.contrib.tpu.python.tpu import keras_support


# python3 Deploy.py <file.hic> </dir/models> </dir/working> <stem> <res,> <norm> <threshold>

def convert_models_for_tpu(all_model_lists):
    all_new_model_list = []
    for model_list in all_model_lists:
        new_model_list = []
        for model0 in model_list:
            tpu_grpc_url = "grpc://" + os.environ["COLAB_TPU_ADDR"]
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
            strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
            model1 = tf.contrib.tpu.keras_to_tpu_model(model0, strategy=strategy)
            new_model_list.append(model1)
        all_new_model_list.append(new_model_list)
    return all_new_model_list


if __name__ == "__main__":
    MATRIX_WIDTH = 500
    FILEPATH = sys.argv[1]
    MODEL_DIR = sys.argv[2]
    WORKING_DIR = sys.argv[3]
    STEM = sys.argv[4]
    RESOLUTIONS = sys.argv[5].split(",")
    NORMALIZATION_TYPE = sys.argv[6]  # "KR"
    THRESHOLD = float(sys.argv[7])
    MAX_EXAMPLES_IN_RAM = 30
    BATCH_SIZE = 30
    NUM_STRAW_WORKERS = 10
    FEATURE_TYPES = ["loops", "domains", "stripes"]
    ALL_MODELS = load_models(FEATURE_TYPES, MODEL_DIR)
    TPU_MODELS = convert_models_for_tpu(ALL_MODELS)
    DeployTridentFeatures(FILEPATH, ALL_MODELS, RESOLUTIONS, STEM, WORKING_DIR, NORMALIZATION_TYPE, THRESHOLD,
                          MATRIX_WIDTH, MAX_EXAMPLES_IN_RAM, NUM_STRAW_WORKERS, BATCH_SIZE)
