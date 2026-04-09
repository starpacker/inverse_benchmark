import numpy as np

import matplotlib

matplotlib.use("Agg")

def forward_operator(z, F_z, k, f0, A, noise_level, seed):
    """
    Compute FM-AFM frequency shift from force using the
    small-amplitude approximation (valid when A << interaction range):

    Δf(d)/f0 ≈ -(1/(2k)) × dF/dz |_{z=d}
    This gives us: Δf(d) = -(f0/(2k)) × F'(d)
    
    Parameters:
    -----------
    z : ndarray
        Distance grid (m)
    F_z : ndarray
        Force curve (N)
    k : float
        Cantilever spring constant (N/m)
    f0 : float
        Resonance frequency (Hz)
    A : float
        Oscillation amplitude (m)
    noise_level : float
        Fractional noise on Δf
    seed : int
        Random seed
        
    Returns:
    --------
    delta_f_noisy : ndarray
        Noisy frequency shift (Hz)
    """
    np.random.seed(seed)
    
    dz = z[1] - z[0]
    dF_dz = np.gradient(F_z, dz)
    delta_f = -(f0 / (2.0 * k)) * dF_dz
    
    # Add noise
    noise = noise_level * np.max(np.abs(delta_f)) * np.random.randn(len(delta_f))
    delta_f_noisy = delta_f + noise
    
    return delta_f_noisy
