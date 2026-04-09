import numpy as np

import matplotlib

matplotlib.use('Agg')

def compute_relative_error(x_true, x_recon):
    return float(np.linalg.norm(x_true - x_recon) / (np.linalg.norm(x_true) + 1e-30))
