import math
import numpy as np

from nnlearn.nanograd import Var
from nnlearn.exceptions import FeatureNotImplemented


class DenseLayer:
    
    """Implementation of a dense layer.

    Dense means that all neurons from previous layer are
    connected with neurons in the following layer.

    Parameters
    ----------
    nin : int
      Number of neurons in the previous layer.

    nout : int
      Number of neurons in the this layer.
    """
  
    def __init__(self, nin, nout, activation='relu'):
        self.nin = nin
        self.nout = nout
        
        # Attributes that need to be explicitly setup
        self.weights = None
        self.bias = None
        self.activation = None
        self.lr = None
        
        # Setup of the attributes
        self._run_setup(activation)

    def step(self, X):
        """Step forward through the dense layer.

        Parameters
        ----------
        X : 2d array
          Each row represents a sample. Each column represents
          feature/neuron.

        Returns
        -------
        res : 2d array
          Each row represents a sample. Each column represents
          feature/neuron. 
        """
        res = self.activation(X @ self.weights + self.bias)
        return res

    def _set_grad_zero(self, x):
        x.grad = 0

    def _zero_grads(self):
        gradzero = np.vectorize(self._set_grad_zero)
        gradzero(self.weights)
        gradzero(self.bias)

    def _update_weight(self, x):
        x.v -= self.lr * x.grad

    def _update_weights(self, lr):
        self.lr = lr
        weight_update = np.vectorize(self._update_weight)
        weight_update(self.weights)
        weight_update(self.bias)

    def _1Dsoftmax(self, a):
        aexp = np.vectorize(lambda x: x.exp())
        tmp = aexp(a)
        return tmp/tmp.sum()
    
    def _2Dsoftmax(self, X):
        return np.apply_along_axis(self._1Dsoftmax, 1, X)

    def _run_setup(self, activation):

        # Setup model's parameters
        self.weights = np.array([[Var(np.random.normal()) for _ in range(self.nout)] for _ in range(self.nin)])
        self.bias = np.array([Var(np.random.normal()) for _ in range(self.nout)])

        # Setup activation function
        if activation == 'relu':
            self.activation = np.vectorize(lambda x: x.relu())
        elif activation  == 'tanh':
            self.activation = np.vectorize(lambda x: x.tanh())
        elif activation == 'softmax':
            self.activation = self._2Dsoftmax
        else:
            text = f"{activation} is not implemented. Choose from ['relu', 'tanh', 'softmax']"
            raise FeatureNotImplementedError(text)

