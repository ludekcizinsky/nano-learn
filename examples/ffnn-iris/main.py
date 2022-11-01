import os 
import sys
sys.path.insert(0, os.path.abspath('../..'))

from nnlearn.network import FFNN, DenseLayer
from nnlearn.datasets import load_iris
from nnlearn.metrics import accuracy_score
from nnlearn.util import ScriptInformation

# TODO: replace this with own implentation
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def test_ffnn_classifier():
    
    logger = ScriptInformation()
    logger.section_start(":herb: FFNN - IRIS")
    logger.script_time()
    logger.author("Ludek", "Cizinsky")
    logger.section_start(":construction: Prepare input for the model")

    logger.working_on("Load and split data") 
    X, y = load_iris()

    logger.working_on("Train test split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    m = X_train.shape[1]

    logger.working_on("Process the data")
    encoder = preprocessing.LabelEncoder().fit(y_train)
    scaler = preprocessing.StandardScaler().fit(X_train)

    y_train = encoder.transform(y_train)
    X_train = scaler.transform(X_train)

    y_test = encoder.transform(y_test)
    X_test = scaler.transform(X_test)
 
    logger.working_on("Define the layers")
    layers = [DenseLayer(m, 3), DenseLayer(3, 4), DenseLayer(4, 3, activation='softmax')]

    logger.section_start(":robot: Train the model")
    figpath = "report/figures/"
    clf = FFNN(layers,
               epochs = 50,
               loss_func='cross_entropy',
               batch_size=.5,
               lr=.9,
               shuffle=True,
               figpath=figpath)
    clf.fit(X_train, y_train)
    logger.c.print(clf.report)

    logger.section_start(":crystal_ball: Validate the model")
    y_hat = clf.predict(X_test)
    acc = accuracy_score(y_test, y_hat)
    logger.important_metric('Accuracy', acc)

    logger.save("report/report.html")

if __name__ == '__main__':
    test_ffnn_classifier()

