from nnlearn.util import _convert_to_np_arr
from nnlearn.exceptions import DimensionMismatchError
from collections import Counter
import numpy as np

def accuracy_score(y_true, y_hat, **kwargs):

    """
    Number of correctly classified records over the number of all records.

    Parameters
    ----------
    y_true : iterable object
        Ground values.
    y_hat : iterale object
        Predicted values.
    **kwargs
        Arbitrary keyword arguments.
    
    Returns
    -------
    float
        Accuracy score.
    
    Notes
    -----
    Accuracy score is a useful metric when your dataset is balanced in terms of labels.
    """


    y_true, y_hat = _convert_to_np_arr(y_true), _convert_to_np_arr(y_hat)

    if y_true.shape[0] != y_hat.shape[0]:
        raise DimensionMismatchError(f'y_true has shape {y_true.shape} and {y_hat.shape}. They both bust be one dimensional and of same length.')

    return (y_true == y_hat).sum()/y_true.shape[0]


def gini_score(y):

    """
    Measure of ``Gini impurity``.

    Parameters
    ----------
    y : 1d array
        Labels of classes.
    
    Returns
    -------
    float
        Float between 0 and 1.

    Notes
    -----
    `Gini impurity` is usually used within the context of
    DecisionTrees. The value ranges between 0 and 1.
    If 0, it means that within your dataset, you only have one class.
    If more than 0, it means that there is certain likelihood that
    you will misclassify given sample from yout dataset.

    For more info, I suggest you visit this `blog <https://bambielli.com/til/2017-10-29-gini-impurity/>`_.
    """

    counts = Counter(y)
    n = len(y)
    result = 0
    for count in counts.values():
        result += (count/n)*(1 - count/n)

    return result


def entropy_score(y):

    """
    Measure of ``Entropy`` of a random variable Y.

    Parameters
    ----------
    y : 1d array
        Labels of classes.
        
    Returns
    -------
    float
        Value between 0 to +inf depending on the number of clasess.
    
    Notes
    -----
    ``Entropy`` is a measure of disorder. The higher the entropy,
    the more disorder there is present. As an example,
    if you have binary classes where 50 % is positive and the
    rest negative, then your entropy would be 1 (high), if
    you only have positive samples, then your entropy is 0. (low)
    The formula for entropy is as follows:

    .. math::
    
        E(S) = \sum_i^c -p_i log_2 p_i

    where ``c`` is number of classes you have.
    """

    ent = 0 
    for c in np.unique(y):
        pi = y[y == c].shape[0] / y.shape[0]
        ent += -pi*np.log2(pi)
   
    return ent

def information_gain_score(x, y):

    """
    Measure of ``Information gain``.

    Parameters
    ----------
    y1 : 1d array
        Labels of classes in the parent node.
    y2 : 1d array
        Labels of classes after the split of the parent node.

    Notes
    -----
    ``Information gain`` tells you how much you can tell about certain
    variable given some other variable. The formula for the information
    gain is as follows:

    .. math::

        IG(X, Y) = E(Y) - E(Y | X)

    where ``E`` refers to :func:`entropy_score`. If the IG is low, it means
    that given ``X``, we know a lot about ``Y``. In other words, the more
    we reduce entropy (disorder) of our target variable ``Y``, the larger
    the information gain is.k
    """
    
    return entropy_score(y1) - entropy_score(y2)

