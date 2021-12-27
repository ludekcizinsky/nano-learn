from util import _convert_to_np_arr
from exceptions import DimensionMismatchError
from collections import Counter

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
    Measure of `Gini impurity`.

    Paramaters
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

    For more info, I suggest you visit this blog: `here <https://bambielli.com/til/2017-10-29-gini-impurity/>`_.
    """

    counts = Counter(y)
    n = len(y)
    result = 0
    for count in counts.values():
        result += (count/n)*(1 - count/n)

    return result
