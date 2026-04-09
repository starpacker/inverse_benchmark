import matplotlib

matplotlib.use('Agg')

import os

import numpy as np

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_metrics(gt, recon):
    """Compute reconstruction quality metrics."""
    gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-12)
    recon_norm = (recon - recon.min()) / (recon.max() - recon.min() + 1e-12)

    # 3D PSNR
    mse = np.mean((gt_norm - recon_norm)**2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-12))

    # 3D correlation coefficient
    gt_flat = gt_norm.ravel()
    recon_flat = recon_norm.ravel()
    cc = np.corrcoef(gt_flat, recon_flat)[0, 1]

    # RMSE
    rmse = np.sqrt(mse)

    # Compute SSIM on central slices (2D SSIM)
    mid = gt.shape[0] // 2
    ssim_xy = structural_similarity(gt_norm[mid, :, :], recon_norm[mid, :, :],
                                     data_range=1.0)
    ssim_xz = structural_similarity(gt_norm[:, mid, :], recon_norm[:, mid, :],
                                     data_range=1.0)
    ssim_yz = structural_similarity(gt_norm[:, :, mid], recon_norm[:, :, mid],
                                     data_range=1.0)
    ssim_avg = (ssim_xy + ssim_xz + ssim_yz) / 3.0

    return {
        'psnr': float(psnr),
        'ssim_xy': float(ssim_xy),
        'ssim_xz': float(ssim_xz),
        'ssim_yz': float(ssim_yz),
        'ssim_avg': float(ssim_avg),
        'cc': float(cc),
        'rmse': float(rmse),
    }
