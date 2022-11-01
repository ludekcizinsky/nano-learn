from nnlearn.nanograd import Var
from nnlearn.metrics import CLF_METRICS, REG_METRICS
from nnlearn.util import ScriptInformation

import numpy as np

class GdBase:
    """
    Base for all parametric models which are optimized via Gradient descent.
    """

    def __init__(self, batch_size, shuffle, loss_func, epochs, lr):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.logger = ScriptInformation()
        self.loss_func_name = loss_func
        self.loss_func = CLF_METRICS.get(loss_func) or REG_METRICS.get(loss_func)
        self.epochs = epochs
        self.lr = lr

        self.Xv = None
        self.yv = None
        self.n = None

        self._f_arr_to_var = np.vectorize(lambda x: Var(x))
        self._f_arr_to_val = np.vectorize(lambda x: x.v)

    def _arr_to_var(self, X):
        """
        Transform given X values into Var objects.

        Notes
        -----
        Assumption made: X is an nd-numpy array with numerical variables.
        """
        
        return self._f_arr_to_var(X)
    
    def _arr_to_val(self, X):
        """
        Get from given X's Var instances their values.
        """
 
        return self._f_arr_to_val(X)

    def _preprocessing(self, X, y=None):

        self.n = X.shape[0]

        # Turn values into Var instances
        if y is not None:
          self.Xv, self.yv = self._arr_to_var(X), self._arr_to_var(y)
        else:
          self.Xv = self._arr_to_var(X)
        
        # If batch size is fraction, turn into actual size
        if isinstance(self.batch_size, float):
            self.batch_size = int(self.n*self.batch_size)
    
    def _get_batches(self): 

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

        if self.shuffle:
            rows_index = np.array([i for i in range(self.n)])
            np.random.shuffle(rows_index) # In-place

            self.Xv = self.Xv[rows_index]
            self.yv = self.yv[rows_index]

