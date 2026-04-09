import numpy as np

import matplotlib

matplotlib.use('Agg')

def compute_ssim_simple(x_true, x_recon):
    mu_x, mu_y = np.mean(x_true), np.mean(x_recon)
    sigma_x, sigma_y = np.std(x_true), np.std(x_recon)
    sigma_xy = np.mean((x_true - mu_x) * (x_recon - mu_y))
    data_range = np.max(x_true) - np.min(x_true)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    return float(((2*mu_x*mu_y+c1)*(2*sigma_xy+c2)) /
                 ((mu_x**2+mu_y**2+c1)*(sigma_x**2+sigma_y**2+c2)))
