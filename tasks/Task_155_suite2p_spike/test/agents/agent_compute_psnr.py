import matplotlib

matplotlib.use('Agg')

import numpy as np

def compute_psnr(clean, noisy):
    """Peak signal-to-noise ratio in dB."""
    mse = np.mean((clean - noisy) ** 2)
    if mse == 0:
        return float('inf')
    peak = np.max(np.abs(clean))
    if peak == 0:
        return 0.0
    return float(10 * np.log10(peak ** 2 / mse))
