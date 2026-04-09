import numpy as np

import matplotlib

matplotlib.use("Agg")

from scipy.ndimage import gaussian_filter1d

def compute_ssim_1d(gt, recon):
    """Compute a 1-D analogue of SSIM."""
    data_range = np.max(gt) - np.min(gt)
    if data_range < 1e-12:
        return 0.0
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    win_sigma = 11.0 / 6.0

    mu_x = gaussian_filter1d(gt, sigma=win_sigma)
    mu_y = gaussian_filter1d(recon, sigma=win_sigma)
    sig_x2 = gaussian_filter1d(gt ** 2, sigma=win_sigma) - mu_x ** 2
    sig_y2 = gaussian_filter1d(recon ** 2, sigma=win_sigma) - mu_y ** 2
    sig_xy = gaussian_filter1d(gt * recon, sigma=win_sigma) - mu_x * mu_y

    sig_x2 = np.maximum(sig_x2, 0)
    sig_y2 = np.maximum(sig_y2, 0)

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x2 + sig_y2 + C2))
    return float(np.mean(ssim_map))
