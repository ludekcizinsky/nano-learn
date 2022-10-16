from nnlearn.tree import DecisionTreeClassifier
from nnlearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from nnlearn.metrics import accuracy_score

def test_dt_classifier():

    print('='*50)
    X, y = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = DecisionTreeClassifier(max_features=3)
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    print(f'Accuracy score: {accuracy_score(y_test, y_hat)}')
    print('='*50)

