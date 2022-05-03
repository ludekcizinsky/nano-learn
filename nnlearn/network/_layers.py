import numpy as np
from ._nanograd import Var


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
  
    def __init__(self, nin, nout):
        self.nin = nin
        self.nout = nout
        self.weights = None
        self.bias = None

        self._run_setup()

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
        res = X@self.weights + self.bias
        return res

    def _zero_grads(self):
        vec = np.vectorize(myfunc)
result = myfunc_vec(mymatrix)

    def _run_setup(self):
        """Initialize the dense layer based on provided info.
        """ 
        self.weights = np.array([[Var(np.random.normal()) for _ in range(nout)] for _ in range(nin)])
        self.bias = np.array([Var(np.random.normal()) for _ in range(nout)])

