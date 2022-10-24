from nnlearn.exceptions import FeatureNotImplemented

import numpy as np

class SVM:
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass
    
    def _gram(self, X, k, how='less_naive'):
        
        """Compute the Gram matrix
        
        For every pair (combination) of vectors in X
        compute the transformation and return corresponding
        value. Therefore entry i,j represents the computed value
        between ith and jth vector.

        Attributes
        ----------
        X : 2d array
            Design matrix where each row represents a sample

        k : function
            Function that takes two m-dimensional vectors as input
            and returns single scalar value output

        how : string
            One of following (naive, less-naive) - see notes for
            more info.

        Notes
        -----
        I implemented two methods:
            - naive: O(N^2)
            - less naive: O(N^2/2) using numpy
        """
    
        if how == 'naive':
            return self._gram_naive(X, k)
        elif how == 'less_naive':
            return self._gram_less_naive(X, k)
        else:
            text = f"{how} has not been implemented, see docs for valid input."
            raise FeatureNotImplemented(text)
 
    @staticmethod
    def _gram_naive(X, k):

        N = X.shape[0]
        K = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                K[i, j] = k(X[i], X[j])

        return K
    
    @staticmethod
    def _gram_less_naive(X, k):

        # Get all combinations of vectors
        n = X.shape[0]
        i, j = np.triu_indices(n, k=0)
        xi = X[i]
        xj = X[j]

        # Compute the value using the given kernel
        result = np.zeros((n, n))
        pairsnum = xi.shape[0]
        result[i,j] = [k(xi[i], xj[i]) for i in range(pairsnum)]
        result = (result + result.T)

        # Adjust the main digonal to count it only once
        di = np.diag_indices(n)
        result[di] = result[di]/2

        return result

