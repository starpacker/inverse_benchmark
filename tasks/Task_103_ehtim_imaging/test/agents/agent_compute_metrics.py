import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

from skimage.metrics import structural_similarity as ssim

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_103_ehtim_imaging"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def compute_metrics(gt, rec):
    """Compute PSNR, SSIM, and cross-correlation metrics."""
    gt_n = gt / gt.max() if gt.max() > 0 else gt
    rec_n = rec / rec.max() if rec.max() > 0 else rec
    mse = np.mean((gt_n - rec_n) ** 2)
    psnr = 10.0 * np.log10(1.0 / mse) if mse > 1e-15 else 100.0
    dr = max(gt_n.max() - gt_n.min(), rec_n.max() - rec_n.min())
    if dr < 1e-15:
        dr = 1.0
    ssim_val = ssim(gt_n, rec_n, data_range=dr)
    gz = gt_n - gt_n.mean()
    rz = rec_n - rec_n.mean()
    d = np.sqrt(np.sum(gz ** 2) * np.sum(rz ** 2))
    cc = np.sum(gz * rz) / d if d > 1e-15 else 0.0
    return float(psnr), float(ssim_val), float(cc)
