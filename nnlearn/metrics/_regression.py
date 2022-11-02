"""
Module that contains functions for evaluation of regression problems. The following
functions are defined:

- :func:`squared_error`
- :func:`mean_squared_error`
- :func:`absolute_error`
- :func:`mean_absolute_error`
"""

import numpy as np
from nnlearn.nanograd import Var

def squared_error(y, p, var):
    """Squared error.

    Sqaured error can be defined as follows:

    .. math::

        \sum_i^n (y_i - p_i)^2
    
    where :math:`n` is the number of provided records.

    Parameters
    ----------
    y : :class:`ndarray`
        One dimensional array with ground truth values.
    
    p : :class:`ndarray`
        One dimensional array with predicted values.

    Returns
    -------
    float
        Squared error as desribed above.
    
    Notes
    -----
    Usually used for regression problems.
    """
    if var:
        return (y - p).sqr()
    else:
        return np.sum((y - p)**2)

def mean_squared_error(Y, P, var=True):
    """Mean of squared error

    Parameters
    ----------
    Y : :class:`ndarray`
        One dimensional array with ground truth values.
    
    P : :class:`ndarray`
        One dimensional array with predicted values.

    Returns
    -------
    float
        Mean squared error.
    """

    if var:
        n = Var(Y.shape[0])
        total = Var(0)
        for i in range(n.v):
          y = Y[i] # true class
          yhat = P[i]
          total += squared_error(y, yhat, var)

        return total/n

    else:
        n = Y.shape[0]
        return squared_error(Y, P, var)

def absolute_error(y, p):

    """Absolute error.

    Absolute error can be defined as follows:

    .. math::

        \sum_i^n abs(y_i - p_i)
    
    where :math:`n` is the number of provided records.

    Parameters
    ----------
    y : :class:`ndarray`
        One dimensional array with ground truth values.
    
    p : :class:`ndarray`
        One dimensional array with predicted values.

    Returns
    -------
    float
        Absolute error as desribed above.
    """
    return np.abs(y-p).sum()

def mean_absolute_error(y, p):
    """Mean absolute error

    Parameters
    ----------
    y : :class:`ndarray`
        One dimensional array with ground truth values.
    
    p : :class:`ndarray`
        One dimensional array with predicted values.

    Returns
    -------
    float
        Mean absolute error as desribed above.
    """

    n = y.shape[0]
    return absolute_error(y, p)/n

# TODO: extend this to all functions
REG_METRICS = {
        "mse": mean_squared_error
}
