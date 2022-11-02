import numpy as np
from rich.progress import track

from nnlearn.reporting import GdReport
from nnlearn.nanograd import Var
from nnlearn.base import GdBase

class FFNN(GdBase, GdReport):

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
    """

    def __init__(self,
        layers,
        loss_func='mse',
        epochs=50,
        batch_size=1.0,
        shuffle=False,
        lr=.01,
        figpath=""):
        
        # Common attributes to models optimized via GD
        GdBase.__init__(self,
                        batch_size,
                        shuffle,
                        loss_func,
                        epochs,
                        lr)

        # Reporting
        GdReport.__init__(self, figpath, 'clf')

        # FFNN specific
        self.layers = layers
 
    def _forward(self, X):
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
        for epoch in track(range(1, self.epochs + 1), "Training..."):
            self._reshuffle()
            X_batches, y_batches = self._get_batches()

            batch = 1
            losses = []
            for X, y in zip(X_batches, y_batches):

                # Predict
                yhat = self._forward(X)

                # Compute loss based on the prediction
                loss = self.loss_func(y, yhat)
                losses.append(loss.v)

                # reset gradients of variables to zero
                self._zero_grads() 

                # backward propagate 
                loss.backward()

                # update weights
                self._update_weights()
                
                # Increase batch number
                batch += 1
            
            # Epoch evaluation
            yhat_train = np.argmax(self.predict_proba(self.Xv), axis=1)
            y_train = self._arr_to_val(self.yv)
            self.eval_epoch(epoch, losses, y_train, yhat_train)

        self.create_report(self.loss_func_name, self.batch_size, self.lr)
 
    def fit(self, X, y):
        self._preprocessing(X, y)
        self._train()

    def predict_proba(self, X):
        return  self._arr_to_val(self._forward(X))

    def predict(self, X):
        Xv = self._arr_to_var(X)
        probs = self.predict_proba(Xv)
        return np.argmax(probs, axis=1)

