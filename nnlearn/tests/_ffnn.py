from nnlearn.network import FFNN, DenseLayer
from nnlearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from nnlearn.metrics import accuracy_score

# TODO: replace this with own implentation
from sklearn import preprocessing

def test_ffnn_classifier():

    # Load and split data
    X, y = load_iris()

    # Process the data
    y = preprocessing.LabelEncoder().fit_transform(y)
    X = preprocessing.StandardScaler().fit_transform(X)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    m = X_train.shape[1]

    # Define the layers
    layers = [DenseLayer(m, 3), DenseLayer(3, 4), DenseLayer(4, 3, activation='softmax')]

    # Train the model 
    clf = FFNN(layers,
               epochs = 100,
               loss_func='cross_entropy',
               batch_size=.5,
               lr=.9,
               shuffle=True)
    clf.fit(X_train, y_train)

    # Validate the model
    y_hat = clf.predict(X_test)
    print(f'Validation accuracy: {accuracy_score(y_test, y_hat)}')

