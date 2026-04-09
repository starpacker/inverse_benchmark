import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_psnr(ref, test, data_range=None):
    """Compute Peak Signal-to-Noise Ratio."""
    if data_range is None:
        data_range = max(ref.max() - ref.min(), 1e-10)
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64))**2)
    if mse < 1e-30:
        return 100.0
    return float(10 * np.log10(data_range**2 / mse))

def compute_rmse(ref, test):
    """Compute Root Mean Square Error."""
    return float(np.sqrt(np.mean((ref - test)**2)))

def evaluate_results(all_dx_recon, all_dy_recon, dx_gt, dy_gt, 
                     grid_ys, grid_xs, ref_image, images, params):
    """
    Evaluate DIC reconstruction results and generate visualizations.
    
    Args:
        all_dx_recon: Recovered x-displacement fields (n_frames, ny, nx)
        all_dy_recon: Recovered y-displacement fields (n_frames, ny, nx)
        dx_gt: Ground truth x-displacement fields (n_frames, height, width)
        dy_gt: Ground truth y-displacement fields (n_frames, height, width)
        grid_ys: Y coordinates of grid points
        grid_xs: X coordinates of grid points
        ref_image: Reference speckle image
        images: Deformed image sequence
        params: Dictionary containing parameters
    
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    n_frames = len(all_dx_recon)
    
    # Extract ground truth at grid points
    all_dx_true = []
    all_dy_true = []
    for t in range(n_frames):
        dx_gt_s = dx_gt[t][np.ix_(grid_ys, grid_xs)]
        dy_gt_s = dy_gt[t][np.ix_(grid_ys, grid_xs)]
        all_dx_true.append(dx_gt_s)
        all_dy_true.append(dy_gt_s)
        
        errs = np.sqrt((all_dx_recon[t] - dx_gt_s)**2 + (all_dy_recon[t] - dy_gt_s)**2)
        print(f"       Frame {t:2d}: max_err={np.max(errs):.4f} px,  "
              f"mean_err={np.mean(errs):.4f} px")

    all_dx_true = np.array(all_dx_true)
    all_dy_true = np.array(all_dy_true)

    # Compute displacement magnitudes
    disp_true = np.sqrt(all_dx_true**2 + all_dy_true**2)
    disp_recon = np.sqrt(all_dx_recon**2 + all_dy_recon**2)

    # Compute metrics
    psnr_val = compute_psnr(disp_true, disp_recon)
    rmse_val = compute_rmse(disp_true, disp_recon)

    flat_t = disp_true.ravel()
    flat_r = disp_recon.ravel()
    if np.std(flat_t) > 1e-10 and np.std(flat_r) > 1e-10:
        cc_val = float(np.corrcoef(flat_t, flat_r)[0, 1])
    else:
        cc_val = 1.0 if np.allclose(flat_t, flat_r) else 0.0

    rmse_dx = compute_rmse(all_dx_true, all_dx_recon)
    rmse_dy = compute_rmse(all_dy_true, all_dy_recon)

    ssim_vals = []
    for t in range(n_frames):
        dr = max(disp_true[t].max() - disp_true[t].min(), 1e-10)
        s = ssim(disp_true[t], disp_recon[t], data_range=dr)
        ssim_vals.append(s)
    ssim_mean = float(np.mean(ssim_vals))

    metrics = {
        "psnr_dB": round(psnr_val, 2),
        "ssim": round(ssim_mean, 4),
        "correlation_coefficient": round(cc_val, 6),
        "rmse_displacement_pixels": round(rmse_val, 6),
        "rmse_dx_pixels": round(rmse_dx, 6),
        "rmse_dy_pixels": round(rmse_dy, 6),
        "n_frames": n_frames,
        "image_size": [params['height'], params['width']],
        "method": "ZNCC_integer_plus_LucasKanade_subpixel_DIC"
    }

    print(f"\n{'=' * 44}")
    print(f"  PSNR     = {psnr_val:.2f} dB")
    print(f"  SSIM     = {ssim_mean:.4f}")
    print(f"  CC       = {cc_val:.6f}")
    print(f"  RMSE     = {rmse_val:.6f} px")
    print(f"  RMSE(dx) = {rmse_dx:.6f} px")
    print(f"  RMSE(dy) = {rmse_dy:.6f} px")
    print(f"{'=' * 44}")

    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] {metrics_path}")

    # ---- Visualization ----
    t_vis = int(np.argmax([np.max(np.abs(dx_gt[t])) for t in range(n_frames)]))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    ax = axes[0, 0]
    ax.imshow(ref_image, cmap='gray')
    ax.set_title('(a) Reference Speckle Image')
    ax.axis('off')

    ax = axes[0, 1]
    ax.imshow(images[t_vis], cmap='gray')
    ax.set_title(f'(b) Deformed Image (frame {t_vis})')
    ax.axis('off')

    ax = axes[0, 2]
    gt_mag = np.sqrt(dx_gt[t_vis]**2 + dy_gt[t_vis]**2)
    im = ax.imshow(gt_mag, cmap='hot')
    plt.colorbar(im, ax=ax, label='|d| (px)', shrink=0.8)
    ax.set_title(f'(c) GT Displacement Magnitude (frame {t_vis})')
    ax.axis('off')

    ax = axes[1, 0]
    ax.imshow(ref_image, cmap='gray', alpha=0.4)
    Y_grid, X_grid = np.meshgrid(grid_ys, grid_xs, indexing='ij')
    scale = 5
    ax.quiver(X_grid, Y_grid,
              all_dx_true[t_vis] * scale, all_dy_true[t_vis] * scale,
              color='blue', alpha=0.8, scale=1, scale_units='xy',
              label='Ground Truth')
    ax.quiver(X_grid + 1, Y_grid + 1,
              all_dx_recon[t_vis] * scale, all_dy_recon[t_vis] * scale,
              color='red', alpha=0.8, scale=1, scale_units='xy',
              label='DIC Recovered')
    ax.set_title(f'(d) Displacement Vectors (frame {t_vis})')
    ax.legend(loc='upper right', fontsize=8)
    ax.axis('off')

    ax = axes[1, 1]
    ax.scatter(disp_true.ravel(), disp_recon.ravel(),
               s=4, alpha=0.3, c='steelblue')
    lim = max(disp_true.max(), disp_recon.max()) * 1.1
    ax.plot([0, lim], [0, lim], 'r--', lw=1, label='Ideal')
    ax.set_xlabel('True |displacement| (px)')
    ax.set_ylabel('Recovered |displacement| (px)')
    ax.set_title(f'(e) True vs Recovered  (CC={cc_val:.4f})')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    err_map = np.sqrt((all_dx_recon[t_vis] - all_dx_true[t_vis])**2 +
                      (all_dy_recon[t_vis] - all_dy_true[t_vis])**2)
    im = ax.imshow(err_map, cmap='hot', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Error (px)', shrink=0.8)
    ax.set_title(f'(f) Displacement Error Map (frame {t_vis})')

    fig.suptitle(
        f"Task 140: pyidi_dic - DIC Displacement Tracking\n"
        f"PSNR={psnr_val:.2f} dB  |  SSIM={ssim_mean:.4f}  |  "
        f"CC={cc_val:.4f}  |  RMSE={rmse_val:.4f} px",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] {fig_path}")

    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), disp_true)
    np.save(os.path.join(RESULTS_DIR, "recon_output.npy"), disp_recon)
    print(f"[SAVE] ground_truth.npy, recon_output.npy")

    return metrics
