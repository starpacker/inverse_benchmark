import warnings

import numpy as np

import matplotlib

matplotlib.use('Agg')

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=FutureWarning)

def forward_operator(x, forward_model):
    """
    Apply the forward convolution operator to a signal.
    
    Parameters
    ----------
    x : numpy.ndarray
        Input signal of shape (dim,).
    forward_model : cuqi.model.LinearModel
        Linear forward convolution model from CUQIpy.
    
    Returns
    -------
    numpy.ndarray
        Convolved signal (predicted observations).
    """
    y_pred = forward_model @ x
    return np.asarray(y_pred)
