import matplotlib

matplotlib.use('Agg')

import os

import json

import numpy as np

import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def evaluate_results(data, result, results_dir=None):
    """
    Compute final metrics, save outputs (metrics JSON, npy arrays, visualization).

    Parameters
    ----------
    data       : dict from load_and_preprocess_data
    result     : dict from run_inversion
    results_dir: str directory to save results (default RESULTS_DIR)

    Returns
    -------
    metrics : dict with PSNR, SSIM, RMSE and other info
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    os.makedirs(results_dir, exist_ok=True)

    phantom = data['phantom']
    recon = result['recon']
    method = result['method']
    psnr = result['psnr']
    ssim_val = result['ssim']
    fbp = result['fbp']
    fp = result['fbp_psnr']
    fs = result['fbp_ssim']
    sp = result['sart_psnr']
    ss = result['sart_ssim']
    gp = result['gs_psnr']
    gss = result['gs_ssim']

    rmse = float(np.sqrt(mean_squared_error(phantom, recon)))
    print(f"\n    BEST: {method} — PSNR={psnr:.2f}, SSIM={ssim_val:.4f}, RMSE={rmse:.6f}")

    # Save metrics
    print("[6] Saving...")
    metrics = {
        "task": "r2gaussian_ct",
        "method": method,
        "PSNR": round(psnr, 2),
        "SSIM": round(ssim_val, 4),
        "RMSE": round(rmse, 6),
        "n_angles": data['n_angles'],
        "noise_level": data['noise_level'],
        "image_size": data['size'],
        "n_gaussians": 800,
        "FBP_PSNR": round(fp, 2),
        "FBP_SSIM": round(fs, 4),
        "SART_PSNR": round(sp, 2),
        "SART_SSIM": round(ss, 4),
        "GS_PSNR": round(gp, 2),
        "GS_SSIM": round(gss, 4),
    }
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(results_dir, "ground_truth.npy"), phantom)
    np.save(os.path.join(results_dir, "reconstruction.npy"), recon)

    # Visualization
    print("[7] Visualization...")
    err = np.abs(phantom - recon)
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for a in ax:
        a.axis('off')

    n_ang = data['n_angles']

    im0 = ax[0].imshow(phantom, cmap='gray', vmin=0, vmax=1)
    ax[0].set_title('Ground Truth\n(Shepp-Logan)', fontsize=12)
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    im1 = ax[1].imshow(fbp, cmap='gray', vmin=0, vmax=1)
    ax[1].set_title(f'FBP ({n_ang} angles)\nPSNR={fp:.1f}dB SSIM={fs:.3f}', fontsize=12)
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    im2 = ax[2].imshow(recon, cmap='gray', vmin=0, vmax=1)
    ax[2].set_title(f'R2-Gaussian CT\nPSNR={psnr:.1f}dB SSIM={ssim_val:.3f}', fontsize=12)
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    im3 = ax[3].imshow(err, cmap='hot', vmin=0, vmax=0.3)
    ax[3].set_title(f'Error Map\nRMSE={rmse:.4f}', fontsize=12)
    plt.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

    plt.suptitle('R2-Gaussian: CT Reconstruction via Gaussian Splatting',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "reconstruction_result.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*60}")
    print(f"DONE — {method}: PSNR={psnr:.2f}, SSIM={ssim_val:.4f}, RMSE={rmse:.6f}")
    print(f"Results: {results_dir}")
    print(f"{'='*60}")

    return metrics
