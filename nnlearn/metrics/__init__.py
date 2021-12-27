"""
The `nnlearn.metric` module includes functions to measure performance of implemented algorithms.
"""

from ._base import accuracy_score, gini_score, entropy_score, information_gain_score

__all__ = [
    'accuracy_score',
    'gini_score',
    'entropy_score',
    'information_gain_score'
]