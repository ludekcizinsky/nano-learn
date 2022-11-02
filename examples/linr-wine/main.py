import os 
import sys
sys.path.insert(0, os.path.abspath('../..'))
import numpy as np

from nnlearn.linear import LinearRegression as LR
from nnlearn.metrics import mean_squared_error
from nnlearn.util import ScriptInformation

# TODO: replace this with own implentation
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

def test_linr_regressor():
    
    logger = ScriptInformation()
    logger.section_start(":grapes: Linear Regression - Wine data")
    logger.script_time()
    logger.author("Ludek", "Cizinsky")
    logger.section_start(":construction: Prepare input for the model")

    logger.working_on("Load and split data") 
    X, _ = load_wine(return_X_y=True)

    logger.working_on("Shuffle the data")
    idx = np.random.choice(X.shape[0], X.shape[0], replace=False)

    logger.working_on("Set X to be alcohol lvl and y color intensity")
    X, y = X[:, [0]], X[:, 9]

    logger.working_on("Train test split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    m = X_train.shape[1]

    logger.working_on("Process the data")
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    logger.section_start(":robot: Train the model")
    figpath = "report/figures/"
    clf = LR(optimizer='gd_backp',
             epochs = 10,
             loss_func='mse',
             batch_size=.5,
             lr=.25,
             shuffle=True,
             bias=True,
             figpath=figpath)
    clf.fit(X_train, y_train)
    logger.c.print(clf.report)

    logger.section_start(":crystal_ball: Validate the model")
    y_hat = clf.predict(X_test)
    mse = mean_squared_error(y_test, y_hat, var=False)
    logger.important_metric('MSE', mse)

    logger.save("report/report.html")

if __name__ == '__main__':
    test_linr_regressor()

