class Error(Exception):
    """
    Base class for exceptions in this module.
    """
    pass

class DimensionMismatchError(Error):
    """
    Mismatch in dimension of two arrays.

    Parameters
    ----------
    message : str
        Explanation of what happened.
    """

    def __init__(self, message):
        self.message = message
    
    def __repr__(self):
        return self.message

class CriterionFunctionNotFound(Error):
    """
    Specified Criterion Function was not found.

    Parameters
    ----------
    message : str
        Explanation of what happened.
    """

    def __init__(self, message):
        self.message = message
    
    def __repr__(self):
        return self.message

class FeatureNotImplemented(Error):
    """
    Given feature has not been implemented.

    Parameters
    ----------
    message : str
        Explanation of what happened.
    """

    def __init__(self, message):
        self.message = message
    
    def __repr__(self):
        return self.message
