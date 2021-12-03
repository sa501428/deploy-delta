import sys
import os
from scripts.DeploymentUtils import DeployTridentFeatures, load_models

if __name__ == "__main__":
    if len(sys.argv) == 8:
        is_insitu = True
    elif len(sys.argv) == 9:
        is_insitu = False
    else:
        print("Invalid number of arguments provided")
        print("Usage:\npython3 Deploy.py <file.hic> </dir/models> </dir/working> <stem> <res,> <norm> "
              "<threshold> [intact]")
        sys.exit(9)
    MATRIX_WIDTH = 500
    FILEPATH = sys.argv[1]
    MODEL_DIR = sys.argv[2]
    OUTPUT_DIR = sys.argv[3]
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    STEM = sys.argv[4]
    RESOLUTIONS = sys.argv[5].split(",")
    NORMALIZATION_TYPE = sys.argv[6]  # "KR"
    THRESHOLD = float(sys.argv[7])
    MAX_EXAMPLES_IN_RAM = 30
    BATCH_SIZE = 30
    NUM_STRAW_WORKERS = 10
    FEATURE_TYPES = ["loops", "domains", "stripes"]
    ALL_MODELS = load_models(FEATURE_TYPES, MODEL_DIR)
    DeployTridentFeatures(FILEPATH, ALL_MODELS, RESOLUTIONS, STEM, OUTPUT_DIR, NORMALIZATION_TYPE, THRESHOLD,
                          MATRIX_WIDTH, MAX_EXAMPLES_IN_RAM, NUM_STRAW_WORKERS, BATCH_SIZE, is_insitu)
