from nnlearn.exceptions import FeatureNotImplemented

import numpy as np
from scipy.optimize import minimize

class SVM:
    """Support vector machine model

    Attributes
    ----------
    kernel : function
             Function that takes two m-dimensional vectors as input
             and returns single scalar value output
    """
    def __init__(self, kernel):
        self.k = kernel

        self.X = None
        self.y = None
        self.K = None
        self.a = None

    def fit(self, X, y):
        """Trains SVM

        Attributes
        ----------
        X : 2D array
            Design matrix, i.e., matrix where each row represents a sample

        y : 1D array
            One dimensional array of size N representing true values
            of our target variable.

        Notes
        -----
        Constraints
          Write the constraints in matrix notation: inequalities should be formulated as f >= 0 , equalities as f = 0
          We have one equality constraint: a @ y.T = 0
          We have N inequalities (one for each a_i), therefore: A @ a >= 0
        """

        # Save the training data to the instance
        self.X, self.y = X, y
        N = X.shape[0]

        # Compute matrix K
        self.K = self._gram()

        # Define constraints 
        A = np.eye(N)
        constraints = ({'type': 'ineq', 'fun': lambda a: A @ a, 'jac': lambda a: A},
                        {'type': 'eq', 'fun': lambda a: a @ y.T, 'jac': lambda a: y.T})

        ## Train
        a0 = np.random.rand(N)  # initial guess
        print('Initial loss: ' + str(self._loss(a0)))

        res = minimize(self._loss, a0, jac=self._get_jac, constraints=constraints, method='SLSQP', options={})
        print('Optimized loss: ' + str(res.fun))
        
        # Optimal Lagrange multipliers
        a = res.x  
        a[np.isclose(a, 0)] = 0  # zero out value that are nearly zeros
        self.a = a

    def predict(self):
        pass

    def _get_jac(self, a):
        """Calculate the Jacobian of the loss function (for the QP solver)

        Attributes
        ----------
        a : 1D array
            One dimensional array of size N where N is the number of samples.
            See notes for details.

        Returns
        -------
        j : 1D array
            Flattened jacobian matrix
        """

        a = a.reshape(1,-1)
        yv = self.y.reshape(-1,1)
        j = - np.ones_like(a) + a @ ((yv @ yv.T) * self.K)
        return j.flatten()

    def _loss(self, a):
        """Return loss

        Attributes
        ----------
        a : 1D array
            One dimensional array of size N where N is the number of samples.
            See notes for details.

        Returns
        -------
        loss : float
               Loss for the given set of parameters a.
        
        Notes
        -----
        TODO: information about the loss function
        """

        # Reshape a and y such that they fit the formula for A
        a = a.reshape(1,-1)
        yv = self.y.reshape(-1,1)

        # Get A
        A = (a.T @ a) * (yv @ yv.T) * self.K

        # Compute the loss
        loss = -(np.sum(a) - 1/2*np.sum(A))
        
        return loss

    
    def _gram(self, how='less_naive'):
        
        """Compute the Gram matrix
        
        For every pair (combination) of vectors in X
        compute the transformation and return corresponding
        value. Therefore entry i,j represents the computed value
        between ith and jth vector.

        Attributes
        ----------
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
            return self._gram_naive()
        elif how == 'less_naive':
            return self._gram_less_naive()
        else:
            text = f"{how} has not been implemented, see docs for valid input."
            raise FeatureNotImplemented(text)
 
    def _gram_naive(self):

        N = self.X.shape[0]
        K = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                K[i, j] = self.k(self.X[i], self.X[j])

        return K
    
    def _gram_less_naive(self):

        # Get all combinations of vectors
        n = self.X.shape[0]
        i, j = np.triu_indices(n, k=0)
        xi = self.X[i]
        xj = self.X[j]

        # Compute the value using the given kernel
        result = np.zeros((n, n))
        pairsnum = xi.shape[0]
        result[i,j] = [self.k(xi[i], xj[i]) for i in range(pairsnum)]
        result = (result + result.T)

        # Adjust the main digonal to count it only once
        di = np.diag_indices(n)
        result[di] = result[di]/2

        return result

