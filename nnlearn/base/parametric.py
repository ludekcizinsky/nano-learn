from nnlearn.nanograd import Var
import numpy as np

class GdBase:
    """
    Base for all parametric models which are optimized via Gradient descent.
    """

    def __init__(self):
      pass
    
    @staticmethod
    def _arr_to_var(X):
        """
        Transform given X values into Var objects.

        Notes
        -----
        Assumption made: X is an nd-numpy array with numerical variables.
        """
        
        f = np.vectorize(lambda x: Var(x))
        return f(X)
    
    @staticmethod
    def _arr_to_val(X):
        """
        Get from given X's Var instances their values.
        """
        f = np.vectorize(lambda x: x.v)
        return f(X)

