import numpy as np

import matplotlib

matplotlib.use("Agg")

def forward_operator(spatial_modes, discrete_eigenvalues, amplitudes, nt):
    """
    Forward model for DMD: reconstruct spatiotemporal data from modes.
    
    The DMD forward model is:
        X(x, t) = Σ_i φ_i(x) · μ_i^k · b_i
    where:
        φ_i are spatial modes
        μ_i are discrete eigenvalues (μ = exp((σ + jω)·dt))
        b_i are mode amplitudes
        k is the time index
    
    Parameters
    ----------
    spatial_modes : ndarray (n_spatial, n_modes)
        DMD spatial modes (complex)
    discrete_eigenvalues : ndarray (n_modes,)
        Discrete eigenvalues from DMD
    amplitudes : ndarray (n_modes,)
        Mode amplitudes (b coefficients)
    nt : int
        Number of time steps to reconstruct
    
    Returns
    -------
    y_pred : ndarray (n_spatial, nt)
        Reconstructed spatiotemporal field
    """
    n_spatial = spatial_modes.shape[0]
    n_modes = len(discrete_eigenvalues)
    
    # Build Vandermonde matrix for temporal evolution
    k = np.arange(nt)
    vander = np.zeros((n_modes, nt), dtype=complex)
    for i in range(n_modes):
        vander[i, :] = discrete_eigenvalues[i] ** k
    
    # Compute dynamics: each row is b_i * μ_i^k
    dynamics = np.diag(amplitudes) @ vander
    
    # Reconstruct: X = Φ @ dynamics
    y_pred = spatial_modes @ dynamics
    
    return y_pred.real
