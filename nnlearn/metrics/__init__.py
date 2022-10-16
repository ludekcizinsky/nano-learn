"""
The `nnlearn.metric` module includes functions to measure performance of implemented algorithms.
"""

from ._classification import *
from ._regression import *

__all__ = [
    'accuracy_score',
    'gini_score',
    'entropy_score',
    'cross_entropy_score',
    'mean_cross_entropy_score',
    'information_gain_score',
    'error_rate',

    'squared_error',
    'mean_squared_error',
    'absolute_error',
    'mean_absolute_error'
]
