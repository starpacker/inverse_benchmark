import numpy as np

import matplotlib

matplotlib.use('Agg')

def forward_operator(sources, mixing_matrix, noise_std=0.0):
    """
    Apply the forward model: y = A @ s + noise
    
    Parameters
    ----------
    sources : np.ndarray
        Source signals, shape (n_sources, N).
    mixing_matrix : np.ndarray
        Mixing matrix A, shape (n_sensors, n_sources).
    noise_std : float
        Standard deviation of additive Gaussian noise.
    
    Returns
    -------
    y_pred : np.ndarray
        Predicted mixed observations, shape (n_sensors, N).
    """
    y_pred = mixing_matrix @ sources
    if noise_std > 0:
        y_pred = y_pred + noise_std * np.random.randn(*y_pred.shape)
    return y_pred
