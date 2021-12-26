import numpy as np

def _convert_to_np_arr(arr):

    if type(arr) == np.ndarray:
        return arr
    else:
        return np.array(arr)
