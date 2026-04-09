import numpy as np

import matplotlib

matplotlib.use('Agg')

def compute_psnr(gt, recon):
    """Compute PSNR."""
    mse = np.mean((gt - recon)**2)
    if mse < 1e-15:
        return 100.0
    data_range = gt.max() - gt.min()
    return 10.0 * np.log10(data_range**2 / mse)
