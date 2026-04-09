import numpy as np

import matplotlib

matplotlib.use("Agg")

def forward_operator(psi, continuum, dt):
    """
    Forward model: compute emission line from transfer function and continuum.
    
    L(t) = ∫ Ψ(τ) × C(t-τ) dτ
    
    Implemented as discrete convolution.
    
    Parameters:
    -----------
    psi : np.ndarray
        Transfer function Ψ(τ)
    continuum : np.ndarray
        Continuum light curve C(t)
    dt : float
        Time step in days
    
    Returns:
    --------
    line_pred : np.ndarray
        Predicted emission line light curve
    """
    n_time = len(continuum)
    line_pred = np.convolve(continuum, psi * dt, mode='full')[:n_time]
    return line_pred
