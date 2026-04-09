import matplotlib

matplotlib.use("Agg")

import numpy as np

from scipy.signal import fftconvolve

def forward_operator(
    time: np.ndarray,
    irf: np.ndarray,
    a1: float,
    tau1: float,
    a2: float,
    tau2: float,
    bg: float,
    total_counts: int
) -> np.ndarray:
    """
    Compute the forward model: IRF convolved with bi-exponential decay plus background.
    
    F(t) = IRF(t) ⊛ [ a₁·exp(-t/τ₁) + a₂·exp(-t/τ₂) ] + background
    
    Parameters
    ----------
    time : np.ndarray
        Time axis (ns).
    irf : np.ndarray
        Normalized instrument response function.
    a1 : float
        Amplitude for first exponential component.
    tau1 : float
        Lifetime for first component (ns).
    a2 : float
        Amplitude for second exponential component.
    tau2 : float
        Lifetime for second component (ns).
    bg : float
        Background counts per bin.
    total_counts : int
        Total photon counts for normalization.
        
    Returns
    -------
    np.ndarray
        Model prediction (counts per bin).
    """
    # Bi-exponential decay
    decay = a1 * np.exp(-time / tau1) + a2 * np.exp(-time / tau2)
    
    # Convolve with IRF
    convolved = fftconvolve(irf, decay, mode="full")[:len(time)]
    
    # Normalize to total counts
    convolved = convolved / convolved.sum() * total_counts
    
    # Add background
    convolved += bg
    
    return convolved
