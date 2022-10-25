import os 
import sys
sys.path.insert(0, os.path.abspath('../..'))

from nnlearn.util import ScriptInformation
from nnlearn.svm import SVM

import numpy as np


def generate_data():

    """Generate random dataset for classification purposes

    Todo
    ----
    Move this to a separate module
    """

    # To make things repeatable
    np.random.seed(1)
    
    # Set size of the sample
    N = 20

    # Generate features
    # * sample two features from gaussian distribution 
    X = np.array(list(zip(np.random.normal(size=N), np.random.normal(size=N))))

    # * indices of points (x,y) for which XOR(x,y) is true - needed for y
    c0 = np.where(X[:,0] < 0)
    # * indices of data points belonging to the other class - needed for y
    c1 = np.where(X[:,0] > 0)  
    
    # * apply rotation on the dataset
    theta = np.radians(45)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c,-s],[s,c]]) # rotation matrix
    X =  X @ R

    # Get targets
    y = np.ones(N)  # labels
    y[c1] = -1  # negative class labels
    X[c1,0] = X[c1,0] + 0.1  # make a little gap between the two groups 

    return X, y

def test_svm_classifier():
    
    # Get data
    X, y = generate_data()

    # Initiliaze logger
    s = ScriptInformation()
    
    # Define kernel function (I chose simple dot product)
    k = lambda x, y: np.dot(x, y)

    # Initialize model
    clf = SVM(kernel=k)
    
    # -- TODO: delete this in next commit
    # Example showing how to use loss function
    # * initial random parameters
    a0 = np.random.rand(X.shape[0])
    initial_loss = clf._loss(X, a0, y)
    print(initial_loss)
    # -- End of the section to be deleted

if __name__ == '__main__':
    test_svm_classifier()

