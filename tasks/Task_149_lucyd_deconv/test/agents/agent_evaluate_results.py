import os

import json

import numpy as np

from skimage.metrics import peak_signal_noise_ratio as psnr_metric

from skimage.metrics import structural_similarity as ssim_metric

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

def evaluate_results(ground_truth, reconstruction, blurred_noisy, params):
    """
    Evaluate reconstruction quality and save results.
    
    Parameters
    ----------
    ground_truth : ndarray
        Ground truth image
    reconstruction : ndarray
        Reconstructed image
    blurred_noisy : ndarray
        Degraded observation
    params : dict
        Parameters used in reconstruction
        
    Returns
    -------
    metrics : dict
        Dictionary containing PSNR, SSIM, RMSE and other metrics
    """
    data_range = 1.0

    # Clip to valid range
    gt = np.clip(ground_truth, 0, data_range)
    recon = np.clip(reconstruction, 0, data_range)
    degraded = np.clip(blurred_noisy, 0, data_range)

    # Compute metrics for reconstruction
    psnr_val = psnr_metric(gt, recon, data_range=data_range)
    ssim_val = ssim_metric(gt, recon, data_range=data_range)
    rmse_val = np.sqrt(np.mean((gt - recon)**2))

    # Compute metrics for degraded image
    degraded_psnr = psnr_metric(gt, degraded, data_range=data_range)
    degraded_ssim = ssim_metric(gt, degraded, data_range=data_range)

    metrics = {
        'PSNR': float(round(psnr_val, 2)),
        'SSIM': float(round(ssim_val, 4)),
        'RMSE': float(round(rmse_val, 4)),
        'degraded_PSNR': float(round(degraded_psnr, 2)),
        'degraded_SSIM': float(round(degraded_ssim, 4)),
        'PSNR_improvement': float(round(psnr_val - degraded_psnr, 2)),
        'method': 'Richardson-Lucy with TV regularization',
        'n_iterations': 150,
        'tv_weight': 0.002,
        'psf_sigma': params.get('psf_sigma', 2.5),
        'photon_gain': params.get('photon_gain', 500)
    }

    # Save results
    np.save(os.path.join(RESULTS_DIR, 'ground_truth.npy'), ground_truth)
    np.save(os.path.join(RESULTS_DIR, 'reconstruction.npy'), reconstruction)

    with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics
