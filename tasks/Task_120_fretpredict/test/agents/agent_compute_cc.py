import numpy as np

import matplotlib

matplotlib.use("Agg")

def compute_cc(gt, recon):
    g = gt - np.mean(gt)
    r = recon - np.mean(recon)
    denom = np.sqrt(np.sum(g ** 2) * np.sum(r ** 2))
    if denom < 1e-15:
        return 0.0
    return float(np.sum(g * r) / denom)
