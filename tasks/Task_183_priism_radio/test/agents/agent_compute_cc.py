import numpy as np

import matplotlib

matplotlib.use('Agg')

def compute_cc(gt, recon):
    """Pearson correlation coefficient."""
    g = gt.ravel()
    r = recon.ravel()
    g_m = g - g.mean()
    r_m = r - r.mean()
    num = np.sum(g_m * r_m)
    den = np.sqrt(np.sum(g_m ** 2) * np.sum(r_m ** 2))
    if den < 1e-15:
        return 0.0
    return float(num / den)
