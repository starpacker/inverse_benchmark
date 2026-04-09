import matplotlib

matplotlib.use('Agg')

import numpy as np

def compute_psnr(gt, recon):
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((gt - recon) ** 2)
    if mse == 0:
        return float('inf')
    data_range = np.max(gt) - np.min(gt)
    return 20.0 * np.log10(data_range / np.sqrt(mse))
