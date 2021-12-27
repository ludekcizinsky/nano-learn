import pandas as pd
import numpy as np
import os
from nnlearn.config.definitions import ROOT_DIR

def load_iris(*args, **kwargs):

    """
    Loads iris data to memory.

    See more about iris dataset `here <https://archive.ics.uci.edu/ml/datasets/iris>`_.

    Parameters
    ----------
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    X : 2darray
        Feature dataset.
    y : 1darray
        Ground truth values.
    """
    
    filepath = os.path.join(ROOT_DIR, 'datasets', 'data','iris.data')
    raw = pd.read_csv(filepath, header=0).to_numpy()
    X = raw[:, :-1].astype(float)
    y = raw[:, -1].astype(str)

    d = {'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}
    new_y = np.copy(y)
    for k, v in d.items(): new_y[y==k] = v
    y = new_y

    return X, y