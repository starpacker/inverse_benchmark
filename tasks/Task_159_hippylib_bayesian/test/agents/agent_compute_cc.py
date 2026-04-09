import numpy as np

import matplotlib

matplotlib.use('Agg')

def compute_cc(x_true, x_recon):
    return float(np.corrcoef(x_true.ravel(), x_recon.ravel())[0, 1])
