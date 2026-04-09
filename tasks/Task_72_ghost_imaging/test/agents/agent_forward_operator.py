import numpy as np

import matplotlib

matplotlib.use('Agg')

def forward_operator(Phi, x, noise_snr_db=None, rng=None):
    """
    Bucket detector measurement: b = Φ @ x + noise.
    
    Args:
        Phi: measurement matrix (M x N)
        x: vectorized image (N,)
        noise_snr_db: optional SNR for adding noise
        rng: random generator for noise
    
    Returns:
        y_pred: predicted bucket measurements (M,)
    """
    b = Phi @ x
    
    if noise_snr_db is not None and rng is not None:
        sig_power = np.mean(b**2)
        noise_power = sig_power / (10**(noise_snr_db / 10))
        noise = np.sqrt(noise_power) * rng.standard_normal(len(b))
        b = b + noise
    
    return b
