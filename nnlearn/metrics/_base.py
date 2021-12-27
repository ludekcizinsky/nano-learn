from nnlearn.util import _convert_to_np_arr
from nnlearn.exceptions import DimensionMismatchError

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


