import numpy as np

import matplotlib

matplotlib.use("Agg")

def compute_rmse(gt, recon):
    return float(np.sqrt(np.mean((gt - recon) ** 2)))
