import numpy as np

import matplotlib

matplotlib.use('Agg')

def forward_operator(q, G, snr_db):
    """
    Forward operator: Source distribution -> Cross-Spectral Matrix (CSM)
    
    Implements: C = G @ diag(q) @ G^H + σ²I
    
    Parameters
    ----------
    q : ndarray
        Source distribution (n_grid,)
    G : ndarray
        Steering vector matrix (n_mics, n_grid)
    snr_db : float
        Signal-to-noise ratio in dB
        
    Returns
    -------
    C : ndarray
        Cross-spectral matrix (n_mics, n_mics), complex Hermitian
    """
    # Signal CSM: C_signal = G @ diag(q) @ G^H
    C_signal = G @ np.diag(q) @ G.conj().T
    
    # Compute noise power based on SNR
    sig_power = np.real(np.trace(C_signal)) / G.shape[0]
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    
    # Add noise to diagonal
    C = C_signal + noise_power * np.eye(G.shape[0])
    
    return C
