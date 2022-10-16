"""
The `nnlearn.exceptions` module provides custom defined errors which can occur during execution.
"""

from ._base import DimensionMismatchError, CriterionFunctionNotFound, FeatureNotImplemented

__all__ = [
    'DimensionMismatchError',
    'CriterionFunctionNotFound',
    'FeatureNotImplemented'
]
