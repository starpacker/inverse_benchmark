import matplotlib

matplotlib.use('Agg')

import numpy as np

def compute_rmse(gt, recon):
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((gt - recon) ** 2)))
