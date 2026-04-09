import numpy as np

import matplotlib

matplotlib.use('Agg')

def _rescale_to_gt(gt, est):
    """Rescale each estimated source to best-fit GT in the least-squares sense."""
    out = np.zeros_like(est)
    for i in range(gt.shape[0]):
        alpha = np.dot(gt[i], est[i]) / (np.dot(est[i], est[i]) + 1e-12)
        out[i] = alpha * est[i]
    return out
