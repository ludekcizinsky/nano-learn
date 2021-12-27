"""
The `nnlearn.exceptions` module provides custom defined errors which can occur during execution.
"""

from ._base import DimensionMismatchError, CriterionFunctionNotFound

__all__ = [
    'DimensionMismatchError',
    'CriterionFunctionNotFound'
]