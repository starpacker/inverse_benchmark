import numpy as np

import matplotlib

matplotlib.use('Agg')

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_metrics_linear(gt_map_2d, recon_map_2d):
    """Compute PSNR/SSIM on normalized linear-scale maps."""
    gt_max = np.max(gt_map_2d)
    if gt_max <= 0:
        gt_max = 1.0

    gt_n = gt_map_2d / gt_max
    recon_max = np.max(recon_map_2d)
    if recon_max <= 0:
        recon_n = np.zeros_like(recon_map_2d)
    else:
        recon_n = recon_map_2d / recon_max

    # Scale recon to minimize MSE (optimal scaling)
    scale = np.sum(gt_n * recon_n) / (np.sum(recon_n**2) + 1e-30)
    recon_scaled = np.clip(recon_n * scale, 0, 1)

    psnr = peak_signal_noise_ratio(gt_n, recon_scaled, data_range=1.0)
    ssim = structural_similarity(gt_n, recon_scaled, data_range=1.0)
    return psnr, ssim
