import numpy as np

import matplotlib

matplotlib.use('Agg')

def compute_psnr(ref, est):
    """Peak SNR (dB) for 1-D signal."""
    mse = np.mean((ref - est) ** 2)
    if mse < 1e-15:
        return 100.0
    peak = np.max(np.abs(ref))
    return 10.0 * np.log10(peak ** 2 / mse)
