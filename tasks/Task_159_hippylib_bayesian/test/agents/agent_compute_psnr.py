import numpy as np

import matplotlib

matplotlib.use('Agg')

def compute_psnr(x_true, x_recon):
    mse = np.mean((x_true - x_recon)**2)
    if mse < 1e-30:
        return 100.0
    data_range = np.max(x_true) - np.min(x_true)
    return 20.0 * np.log10(data_range / np.sqrt(mse))
