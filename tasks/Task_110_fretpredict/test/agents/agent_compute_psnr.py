import numpy as np

import matplotlib

matplotlib.use("Agg")

def compute_psnr(gt, recon):
    mse = np.mean((gt - recon) ** 2)
    if mse < 1e-15:
        return 100.0
    peak = np.max(gt)
    if peak < 1e-12:
        return 0.0
    return float(10 * np.log10(peak ** 2 / mse))
