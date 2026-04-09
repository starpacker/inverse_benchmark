import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import sys

import json

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")

if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

def evaluate_results(x_true, x_reconstructed, y_observed, config, results_dir):
    """
    Compute metrics, visualize, and save results.
    
    Parameters:
    -----------
    x_true : np.ndarray
        Ground truth image (2D)
    x_reconstructed : np.ndarray
        Reconstructed image (2D)
    y_observed : np.ndarray
        Observed data (1D flattened)
    config : dict
        Configuration parameters (lambda_tv, blur_sigma, noise_level, max_iter, img_size)
    results_dir : str
        Directory to save results
        
    Returns:
    --------
    metrics : dict
        Dictionary containing PSNR, SSIM, RMSE, correlation coefficient
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Compute PSNR
    def compute_psnr(ref, test, data_range=None):
        ref = ref.astype(np.float64).ravel()
        test = test.astype(np.float64).ravel()
        if data_range is None:
            data_range = ref.max() - ref.min()
        if data_range < 1e-10:
            data_range = 1.0
        mse = np.mean((ref - test) ** 2)
        if mse < 1e-30:
            return 100.0
        return 10 * np.log10(data_range ** 2 / mse)
    
    # Compute SSIM
    def compute_ssim(ref, test):
        from skimage.metrics import structural_similarity as ssim
        r = ref.squeeze()
        t = test.squeeze()
        data_range = r.max() - r.min()
        if data_range < 1e-10:
            data_range = 1.0
        return float(ssim(r, t, data_range=data_range))
    
    # Compute RMSE
    def compute_rmse(ref, test):
        return float(np.sqrt(np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)))
    
    # Compute Correlation Coefficient
    def compute_correlation(ref, test):
        r = ref.flatten().astype(np.float64)
        t = test.flatten().astype(np.float64)
        r_c = r - r.mean()
        t_c = t - t.mean()
        num = np.sum(r_c * t_c)
        den = np.sqrt(np.sum(r_c**2) * np.sum(t_c**2))
        if den < 1e-30:
            return 0.0
        return float(num / den)
    
    # Compute observation PSNR
    obs_psnr = compute_psnr(x_true.ravel(), y_observed)
    print(f"  Observation PSNR: {obs_psnr:.2f} dB (degraded)")
    
    # Compute metrics
    metrics = {
        "psnr": float(compute_psnr(x_true, x_reconstructed)),
        "ssim": float(compute_ssim(x_true, x_reconstructed)),
        "rmse": float(compute_rmse(x_true, x_reconstructed)),
        "cc": float(compute_correlation(x_true, x_reconstructed)),
        "observation_psnr": float(obs_psnr),
        "solver": "Condat-Vu (primal-dual splitting)",
        "regularizer": "Total Variation (anisotropic, L1 of gradient)",
        "lambda_tv": config["lambda_tv"],
        "blur_sigma": config["blur_sigma"],
        "noise_level": config["noise_level"],
        "max_iterations": config["max_iter"],
        "image_size": config["img_size"],
    }
    
    print(f"  PSNR = {metrics['psnr']:.4f} dB")
    print(f"  SSIM = {metrics['ssim']:.6f}")
    print(f"  RMSE = {metrics['rmse']:.6f}")
    print(f"  CC   = {metrics['cc']:.4f}")
    
    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics -> {metrics_path}")
    
    # Save arrays
    np.save(os.path.join(results_dir, "ground_truth.npy"), x_true)
    np.save(os.path.join(results_dir, "reconstruction.npy"), x_reconstructed)
    np.save("gt_output.npy", x_true)
    np.save("recon_output.npy", x_reconstructed)
    print(f"  Arrays saved")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    gt = x_true.squeeze()
    obs = y_observed.reshape(x_true.shape).squeeze()
    recon = x_reconstructed.squeeze()
    error = np.abs(gt - recon)

    vmin, vmax = 0, 1

    im0 = axes[0, 0].imshow(gt, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('(a) Ground Truth', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(obs, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'(b) Observation (blur σ={config["blur_sigma"]}, noise={config["noise_level"]})',
                         fontsize=13, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(recon, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(f'(c) Reconstruction (PSNR={metrics["psnr"]:.2f} dB)',
                         fontsize=13, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    im3 = axes[1, 0].imshow(error, cmap='hot', vmin=0, vmax=max(error.max(), 0.01))
    axes[1, 0].set_title(f'(d) |Error| (RMSE={metrics["rmse"]:.4f})',
                         fontsize=13, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    mid_row = gt.shape[0] // 2
    axes[1, 1].plot(gt[mid_row, :], 'b-', label='GT', linewidth=2)
    axes[1, 1].plot(obs[mid_row, :], 'g--', label='Observed', linewidth=1, alpha=0.5)
    axes[1, 1].plot(recon[mid_row, :], 'r-', label='Recon', linewidth=1.5)
    axes[1, 1].set_title(f'(e) Profile at row {mid_row}', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].set_xlabel('Column')
    axes[1, 1].set_ylabel('Intensity')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis('off')
    try:
        import pyxu
        pyxu_ver = pyxu.__version__
    except:
        pyxu_ver = "unknown"
    metrics_text = (
        f"Reconstruction Metrics\n"
        f"{'='*30}\n\n"
        f"PSNR:  {metrics['psnr']:.2f} dB\n"
        f"SSIM:  {metrics['ssim']:.4f}\n"
        f"RMSE:  {metrics['rmse']:.6f}\n"
        f"CC:    {metrics['cc']:.4f}\n\n"
        f"{'='*30}\n"
        f"Solver: Condat-Vu (primal-dual)\n"
        f"Library: Pyxu {pyxu_ver}\n"
        f"lambda_TV: {config['lambda_tv']}\n"
        f"Blur sigma: {config['blur_sigma']}\n"
        f"Noise: {config['noise_level']}\n"
        f"Image: {config['img_size']}x{config['img_size']}\n"
        f"Max iter: {config['max_iter']}"
    )
    axes[1, 2].text(0.1, 0.95, metrics_text, transform=axes[1, 2].transAxes,
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle('Pyxu Image Deconvolution: TV-Regularized Proximal Algorithm',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization -> {vis_path}")
    
    return metrics
