import os 
import sys
sys.path.insert(0, os.path.abspath('../../'))

from nnlearn.tree import DecisionTreeClassifier
from nnlearn.datasets import load_iris
from nnlearn.metrics import accuracy_score

#TODO: implement own method
from sklearn.model_selection import train_test_split

def test_dt_classifier():

    print('='*50)
    X, y = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = DecisionTreeClassifier(max_features=3)
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    print(f'Accuracy score: {accuracy_score(y_test, y_hat)}')
    print('='*50)

if __name__ == '__main__':
    test_dt_classifier()

