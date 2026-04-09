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

def evaluate_results(ground_truth, reconstruction, measurements, results_dir):
    """
    Evaluate reconstruction quality and generate visualizations.
    
    Parameters:
        ground_truth: tuple (tx_gt, ty_gt) ground truth traction fields in Pa
        reconstruction: tuple (tx_rec, ty_rec) reconstructed traction fields in Pa
        measurements: tuple (u_meas, v_meas) measured displacement fields in pixels
        results_dir: str, directory to save results
    
    Returns:
        metrics: dict containing all computed metrics
    """
    tx_gt, ty_gt = ground_truth
    tx_rec, ty_rec = reconstruction
    u_meas, v_meas = measurements
    
    # Helper functions for metrics
    def compute_psnr(ref, test, data_range=None):
        """Compute PSNR (dB) between reference and test arrays."""
        if data_range is None:
            data_range = ref.max() - ref.min()
        if data_range == 0:
            return float('inf') if np.allclose(ref, test) else 0.0
        mse = np.mean((ref.astype(float) - test.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10(data_range ** 2 / mse)

    def compute_ssim(ref, test):
        """Compute SSIM for 2D fields."""
        from skimage.metrics import structural_similarity as ssim
        data_range = ref.max() - ref.min()
        if data_range == 0:
            data_range = 1.0
        return ssim(ref, test, data_range=data_range)

    def compute_rmse(ref, test):
        """Compute RMSE."""
        return np.sqrt(np.mean((ref.astype(float) - test.astype(float)) ** 2))

    def compute_relative_error(ref, test):
        """Compute relative error (RE) = ||ref - test||_2 / ||ref||_2."""
        ref_norm = np.linalg.norm(ref.ravel())
        if ref_norm == 0:
            return float('inf')
        return np.linalg.norm((ref - test).ravel()) / ref_norm

    def compute_correlation_coefficient(ref, test):
        """Compute Pearson correlation coefficient."""
        ref_flat = ref.ravel()
        test_flat = test.ravel()
        if np.std(ref_flat) == 0 or np.std(test_flat) == 0:
            return 0.0
        return float(np.corrcoef(ref_flat, test_flat)[0, 1])

    # Compute metrics on traction magnitude
    gt_mag = np.sqrt(tx_gt**2 + ty_gt**2)
    rec_mag = np.sqrt(tx_rec**2 + ty_rec**2)

    metrics = {
        "psnr": float(compute_psnr(gt_mag, rec_mag)),
        "ssim": float(compute_ssim(gt_mag, rec_mag)),
        "rmse": float(compute_rmse(gt_mag, rec_mag)),
        "relative_error": float(compute_relative_error(gt_mag, rec_mag)),
        "correlation_coefficient": float(compute_correlation_coefficient(gt_mag, rec_mag)),
        # Also compute component-wise metrics
        "psnr_tx": float(compute_psnr(tx_gt, tx_rec)),
        "psnr_ty": float(compute_psnr(ty_gt, ty_rec)),
        "rmse_tx": float(compute_rmse(tx_gt, tx_rec)),
        "rmse_ty": float(compute_rmse(ty_gt, ty_rec)),
    }

    print(f"[EVAL] PSNR (magnitude) = {metrics['psnr']:.4f} dB")
    print(f"[EVAL] SSIM (magnitude) = {metrics['ssim']:.6f}")
    print(f"[EVAL] RMSE (magnitude) = {metrics['rmse']:.4f} Pa")
    print(f"[EVAL] Relative Error   = {metrics['relative_error']:.6f}")
    print(f"[EVAL] Correlation Coef = {metrics['correlation_coefficient']:.6f}")

    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics → {metrics_path}")

    # Visualization
    disp_mag = np.sqrt(u_meas**2 + v_meas**2)
    err_mag = np.abs(gt_mag - rec_mag)

    vmax = max(gt_mag.max(), rec_mag.max())

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # Row 1: Scalar magnitude fields
    im0 = axes[0, 0].imshow(gt_mag, cmap='hot', vmin=0, vmax=vmax)
    axes[0, 0].set_title("Ground Truth |T| (Pa)", fontsize=12)
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(disp_mag, cmap='viridis')
    axes[0, 1].set_title("Measured |u| (pixels)", fontsize=12)
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(rec_mag, cmap='hot', vmin=0, vmax=vmax)
    axes[0, 2].set_title("Reconstructed |T| (Pa)", fontsize=12)
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    im3 = axes[0, 3].imshow(err_mag, cmap='magma')
    axes[0, 3].set_title("Error |GT - Recon| (Pa)", fontsize=12)
    plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)

    # Row 2: Quiver plots
    N = tx_gt.shape[0]
    skip = max(1, N // 16)  # subsample for readability
    y_grid, x_grid = np.mgrid[:N, :N]
    sl = (slice(None, None, skip), slice(None, None, skip))

    axes[1, 0].quiver(x_grid[sl], y_grid[sl], tx_gt[sl], ty_gt[sl],
                       gt_mag[sl], cmap='hot', scale_units='xy')
    axes[1, 0].set_title("GT Traction Vectors", fontsize=12)
    axes[1, 0].set_aspect('equal')
    axes[1, 0].invert_yaxis()

    axes[1, 1].quiver(x_grid[sl], y_grid[sl], u_meas[sl], v_meas[sl],
                       disp_mag[sl], cmap='viridis', scale_units='xy')
    axes[1, 1].set_title("Measured Displacement Vectors", fontsize=12)
    axes[1, 1].set_aspect('equal')
    axes[1, 1].invert_yaxis()

    axes[1, 2].quiver(x_grid[sl], y_grid[sl], tx_rec[sl], ty_rec[sl],
                       rec_mag[sl], cmap='hot', scale_units='xy')
    axes[1, 2].set_title("Reconstructed Traction Vectors", fontsize=12)
    axes[1, 2].set_aspect('equal')
    axes[1, 2].invert_yaxis()

    err_tx = tx_gt - tx_rec
    err_ty = ty_gt - ty_rec
    axes[1, 3].quiver(x_grid[sl], y_grid[sl], err_tx[sl], err_ty[sl],
                       err_mag[sl], cmap='magma', scale_units='xy')
    axes[1, 3].set_title("Error Vectors", fontsize=12)
    axes[1, 3].set_aspect('equal')
    axes[1, 3].invert_yaxis()

    fig.suptitle(
        f"pyTFM — Traction Force Microscopy Reconstruction\n"
        f"PSNR={metrics['psnr']:.2f} dB | SSIM={metrics['ssim']:.4f} | "
        f"RMSE={metrics['rmse']:.2f} Pa | RE={metrics['relative_error']:.4f} | "
        f"CC={metrics['correlation_coefficient']:.4f}",
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {vis_path}")

    # Save arrays
    np.save(os.path.join(results_dir, "reconstruction.npy"), np.stack([tx_rec, ty_rec]))
    np.save(os.path.join(results_dir, "ground_truth.npy"), np.stack([tx_gt, ty_gt]))
    np.save(os.path.join(results_dir, "measurements.npy"), np.stack([u_meas, v_meas]))

    return metrics
