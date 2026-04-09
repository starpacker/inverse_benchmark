import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_ssim_2d(gt_2d, pred_2d):
    """Compute SSIM for 2D fields using skimage"""
    try:
        from skimage.metrics import structural_similarity
        data_range = gt_2d.max() - gt_2d.min()
        if data_range < 1e-12:
            data_range = 1.0
        return structural_similarity(gt_2d, pred_2d, data_range=data_range)
    except ImportError:
        mu_gt = np.mean(gt_2d)
        mu_pred = np.mean(pred_2d)
        sig_gt = np.std(gt_2d)
        sig_pred = np.std(pred_2d)
        sig_cross = np.mean((gt_2d - mu_gt) * (pred_2d - mu_pred))
        C1 = (0.01 * (gt_2d.max() - gt_2d.min())) ** 2
        C2 = (0.03 * (gt_2d.max() - gt_2d.min())) ** 2
        ssim = ((2 * mu_gt * mu_pred + C1) * (2 * sig_cross + C2)) / \
               ((mu_gt**2 + mu_pred**2 + C1) * (sig_gt**2 + sig_pred**2 + C2))
        return float(ssim)
