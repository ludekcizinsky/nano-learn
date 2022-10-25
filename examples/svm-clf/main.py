import os 
import sys
sys.path.insert(0, os.path.abspath(''))

from nnlearn.util import ScriptInformation
from nnlearn.svm import SVM

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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


def plot_solution(X, y, clf):
    """Show fitted SWM model
    """
    
    # Define plot
    fig, ax = plt.subplots()

    # Plot training data points
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, ax=ax)

    # Add circles around support vectors
    ax.scatter(clf.X_sv[:, 0], clf.X_sv[:, 1], color='g', s=100, facecolors='none', \
                edgecolors='g', label='support vectors')
    
    # Plot the decision boundary and margins in the input space
    # * get the predictions for the whole grid of points
    grid = np.arange(X.min(), X.max(), 0.05)
    xx, yy = np.meshgrid(grid, grid)
    zz = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    zz = zz.reshape(xx.shape)

    # * Plot only levels at which we have the margins
    # and separating hyperplane 
    CS = plt.contour(xx, yy, zz, levels=[-1, 0, 1])
    plt.clabel(CS, fmt='%2.1d', colors='k');
    plt.xlabel("$x_1$");
    plt.ylabel("$x_2$");
    plt.legend(loc='best');
    plt.title("SVM Classification of Data");

    # Save figure
    fig.savefig("svmclf.png")


def test_svm_classifier():
    
    # Get data
    X, y = generate_data()

    # Initiliaze logger
    s = ScriptInformation()
    
    # Define kernel function (I chose simple dot product)
    k = lambda x, y: np.dot(x, y)

    # Initialize model
    clf = SVM(kernel=k)
    
    # Train the model
    clf.fit(X, y)
    
    # Make a plot of the solution
    plot_solution(X, y, clf)
    
if __name__ == '__main__':
    test_svm_classifier()

