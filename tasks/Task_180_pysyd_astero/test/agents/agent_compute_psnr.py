import numpy as np

import matplotlib

matplotlib.use('Agg')

def compute_psnr(sig, ref):
    """Compute PSNR in log-space."""
    ls = np.log10(np.maximum(sig, 1e-10))
    lr = np.log10(np.maximum(ref, 1e-10))
    mse = np.mean((ls - lr) ** 2)
    if mse < 1e-30:
        return 100.0
    dr = np.max(lr) - np.min(lr)
    return 10.0 * np.log10(dr ** 2 / mse)
