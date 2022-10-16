from nnlearn.nanograd import Var
import numpy as np

class Base:

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
            
        return np.vectorize(lambda x: Var(x))(X)

