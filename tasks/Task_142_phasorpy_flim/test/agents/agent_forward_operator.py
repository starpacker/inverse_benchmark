import matplotlib

matplotlib.use("Agg")

import numpy as np

def forward_operator(
    f1: np.ndarray,
    f2: np.ndarray,
    decay1: np.ndarray,
    decay2: np.ndarray,
    total_photons: int,
) -> np.ndarray:
    """
    Forward model: Generate time-domain FLIM signal from fraction maps.
    
    Computes the expected fluorescence decay at each pixel as a linear
    combination of the two species' decays weighted by their fractions.
    
    Parameters
    ----------
    f1 : np.ndarray
        Fraction map for species 1, shape (nx, ny).
    f2 : np.ndarray
        Fraction map for species 2, shape (nx, ny).
    decay1 : np.ndarray
        Normalized decay profile for species 1, shape (n_time,).
    decay2 : np.ndarray
        Normalized decay profile for species 2, shape (n_time,).
    total_photons : int
        Mean total photon count per pixel.
    
    Returns
    -------
    np.ndarray
        Predicted FLIM signal, shape (nx, ny, n_time).
    """
    signal = (
        f1[:, :, np.newaxis] * decay1[np.newaxis, np.newaxis, :]
        + f2[:, :, np.newaxis] * decay2[np.newaxis, np.newaxis, :]
    ) * total_photons
    
    return signal
