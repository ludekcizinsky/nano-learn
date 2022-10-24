import os 
import sys
sys.path.insert(0, os.path.abspath(''))

from nnlearn.util import ScriptInformation
from nnlearn.svm import SVM

import numpy as np

def test_svm_classifier():
    
    # Initiliaze logger
    s = ScriptInformation()

    # Initialize model
    clf = SVM()

    # -- TODO: delete this in next commit
    # -- Example showing how to use _gram method
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = lambda x, y: np.dot(x, y)
    print()
    print(a)
    print()
    print(clf._gram(a, k, "less_naive"))
    # -- End of the section to be deletd

if __name__ == '__main__':
    test_svm_classifier()

