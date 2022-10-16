import sys
import os
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import warnings
warnings.filterwarnings("ignore")

from tests import test_dt_classifier, test_ffnn_classifier

TESTS = {
    "dt_classifier" : test_dt_classifier,
    "ffnn_classifier" : test_ffnn_classifier
}

if __name__ == "__main__":
    
    TESTS[sys.argv[1]]()

