import numpy as np

from nnlearn.metrics import mean_cross_entropy_score, mean_squared_error, accuracy_score
from nnlearn.nanograd import Var
from ._base import Base


class FFNN(Base):

    """
    Feed forward neural network is a machine learning model which is used for all
    kinds of tasks from regression to classification.

    This class serves as a high level API to setup and train neural network.

    Parameters
    -----------
    layers : iterable
        One dimensional iterable where each item is a :class:`DenseLayer`.
    loss_func : str, optional
        Loss function to use for training.
    epochs : int, optional
        How many times do you want to feed the whole dataset trough the network.
    batch_size : float or int, optional
        Within each epoch, you have the option to feed the network in batches. Here you
        can specify its size.
    shuffle : bool, optional
        Do you want to shuffle the data after each epoch. (Shuffle rows)
    optimizer : str, optional
        Name of the optimizer to tune network's parameters.

    Notes
    -----

    ``Layers``
    Esentially, there are three types of layers - `input, hidden, output.`
    Each layer contains certain number of neurons. The number neurons in the
    input layer is determined based on the number of features in the provided
    dataset. Similarly, in case of output layer, it depends on the task at hand.
    For example, if you are doing regression, then you only need one neuron
    in the output layer or if you are doing multiclass classification, then
    you need same number of neurons as there is distinct classes in the dataset.

    ``Loss functions``
    To be added.

    ``Epoch, batch size``
    To be added.

    ``Training process``
    To be added.

    Todo
    ----

    - Complete notes section

    - Make preprocessing section more robust

    """

    def __init__(self,
        layers,
        loss_func='mse',
        epochs=50,
        batch_size=1.0,
        shuffle=False,
        lr=.01):

        Base.__init__(self)

        self.layers = layers
        self.loss_func = loss_func
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lr = lr

        self.n = None
        self.m = None

        self._run_setup()
    
    def _run_setup(self):

        """
        The goal of this function is to run all neccessary operations
        after client initialies this class.
        """
        
        # Loss functions
        if self.loss_func == 'mse':
            self.loss_func = mean_squared_error
        elif self.loss_func == 'cross_entropy':
            self.loss_func = mean_cross_entropy_score
 
    def _preprocessing(self, X, y=None):

        """ 
        Preprocceses provided training data.

        Attributes
        ----------
        X : 2d array
          Input feature matrix.

        y : 1d array, optional
          Target values

        Notes
        -----
        Following methods are applied:
        - Transform training data into Var object
        - Transform batch size from proportion to actual size if neccessary
        """
        if y is not None:
          self.Xv, self.yv = self._arr_to_var(X), self._arr_to_var(y)
        else:
          self.Xv = self._arr_to_var(X)

        if isinstance(self.batch_size, float):
            self.batch_size = int(self.n*self.batch_size)
    
    def _get_batches(self):

        """
        Splits the given training dataset into the
        batches of given size.

        Returns
        -------
        X_batches : list
            List with batches where each batch is just a subsample of X.
        y_batches : list
            List with batches where each batch is just a subsample of y.
        """
        
        choices = set([i for i in range(self.n)])
        X_batches = []
        y_batches = []
        while len(choices) > 0:

            size = min(self.batch_size, len(choices))
            a = np.array(list(choices))
            selected = np.random.choice(a, size=size, replace=False)

            X_batches.append(self.Xv[selected])
            y_batches.append(self.yv[selected])

            choices = choices - set(selected)
        
        return X_batches, y_batches
    
    def _reshuffle(self):

        """
        Reshuffles the order of rows in the training dataset.
        """

        if self.shuffle:

            rows_index = np.array([i for i in range(self.n)])
            np.random.shuffle(rows_index) # In-place

            self.Xv = self.Xv[rows_index]
            self.yv = self.yv[rows_index]

    def _forward(self, X):
        """Forward step through this neural net.

        Returns
        -------
        res : 2d array
          Each row represents corresponding prediction for given sample.
        """

        res = X
        for l in self.layers:
            res = l.step(res)
        return res
    
    def _zero_grads(self):
        for l in self.layers:
            l._zero_grads()

    def _update_weights(self):
        for l in self.layers:
            l._update_weights(self.lr)

    def _train(self):
        self._reshuffle()
        for epoch in range(1, self.epochs + 1):

            X_batches, y_batches = self._get_batches()

            print(f"Epoch {epoch}")
            print("="*23)
            batch = 1
            losses = []
            for X, y in zip(X_batches, y_batches):

                # Predict
                yhat = self._forward(X)

                # Compute loss based on the prediction
                loss = self.loss_func(y, yhat)
                print(">> Batch {} loss: {:.3f}".format(batch, loss.v))
                losses.append(loss.v)

                # reset gradients of variables to zero
                self._zero_grads() 

                # backward propagate 
                loss.backward()

                # update weights
                self._update_weights()
                
                # Increase batch number
                batch += 1

            print("-"*23) 
            print(">> Mean loss: {:>8.3f}".format(sum(losses)/len(losses)))
            getval = np.vectorize(lambda x: x.v)
            yhat = np.argmax(getval(self._forward(self.Xv)), axis=1)
            ytrue = getval(self.yv)
            print(">> Accuracy: {:>9.3f}".format(accuracy_score(ytrue, yhat)))
            print()
            self._reshuffle()
            
 
    def fit(self, X, y):
        """Find the optimal parameters.

        Parameters
        ----------
        X : 2d array 
          Training data.
        y : 1d array
          Training labels.
        """
        
        self.n, self.m = X.shape
        self._preprocessing(X, y)
        self._train()
 
    def predict(self, X):
        self._preprocessing(X)
        yhat = self._forward(self.Xv)
        getval = np.vectorize(lambda x: x.v)
        return np.argmax(getval(yhat), axis=1)

