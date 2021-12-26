# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

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
    2darray
        Feature dataset.
    1darray
        Ground truth values.
    """

    raw = pd.read_csv('nnlearn/datasets/data/iris.data', header=0).to_numpy()
    X = raw[:, :-1].astype(float)
    y = raw[:, -1].astype(str)

    d = {'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}
    new_y = np.copy(y)
    for k, v in d.items(): new_y[y==k] = v
    y = new_y

    return X, y
