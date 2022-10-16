"""
The module ``nnlearn.network`` includes implementation of feed forward neural
network.
"""

from ._ffnn import FFNN
from ._layers import DenseLayer

__all__ = [
    'FFNN',
    'DenseLayer'
]

