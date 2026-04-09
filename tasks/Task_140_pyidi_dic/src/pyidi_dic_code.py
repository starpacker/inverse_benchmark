"""
pyidi_dic - Digital Image Correlation Displacement Tracking
============================================================
Task 140: From image sequences, track displacement fields using DIC
Repo: https://github.com/ladisk/pyidi

Inverse Problem:
  Forward:  Apply known spatially-varying displacement field to reference
            speckle image -> sequence of deformed images
  Inverse:  From each image pair (ref, deformed), recover the displacement
            field via:
            (a) Coarse integer peak via ZNCC (zero-mean normalised cross-corr)
            (b) Iterative Lucas-Kanade refinement for sub-pixel accuracy

Usage:
    /data/yjh/pyidi_dic_env/bin/python pyidi_dic_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
from skimage.metrics import structural_similarity as ssim

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# =====================================================================
# 1. Generate synthetic speckle pattern
# =====================================================================
def generate_speckle_image(height, width, n_speckles=15000,
                           speckle_sigma=2.5, seed=42):
    """Generate a dense synthetic speckle pattern for DIC."""
    rng = np.random.RandomState(seed)
    img = np.zeros((height, width), dtype=np.float64)

    ys = rng.uniform(0, height, n_speckles)
    xs = rng.uniform(0, width, n_speckles)
    intensities = rng.uniform(0.3, 1.0, n_speckles)

    r = int(4 * speckle_sigma) + 1
    for y0, x0, amp in zip(ys, xs, intensities):
        y_lo = max(0, int(y0) - r)
        y_hi = min(height, int(y0) + r + 1)
        x_lo = max(0, int(x0) - r)
        x_hi = min(width, int(x0) + r + 1)
        yy = np.arange(y_lo, y_hi)[:, None]
        xx = np.arange(x_lo, x_hi)[None, :]
        gauss = amp * np.exp(-((yy - y0)**2 + (xx - x0)**2) /
                              (2 * speckle_sigma**2))
        img[y_lo:y_hi, x_lo:x_hi] += gauss

    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return img


# =====================================================================
# 2. Forward Operator
# =====================================================================
def warp_image(image, dx_field, dy_field):
    """Warp image by displacement field using backward mapping.
    output(r, c) = input(r - dy[r,c], c - dx[r,c])
    """
    h, w = image.shape
    rr, cc = np.meshgrid(np.arange(h, dtype=np.float64),
                         np.arange(w, dtype=np.float64), indexing='ij')
    warped = map_coordinates(image, [rr - dy_field, cc - dx_field],
                             order=3, mode='reflect')
    return warped


def generate_displacement_fields(height, width, n_frames=10):
    """Generate smooth spatially-varying sinusoidal displacement fields."""
    Y, X = np.meshgrid(np.arange(height, dtype=np.float64),
                        np.arange(width, dtype=np.float64), indexing='ij')

    sigma_spatial = min(height, width) / 2.0
    envelope = np.exp(-((X - width / 2)**2 + (Y - height / 2)**2) /
                       (2 * sigma_spatial**2))

    dx_fields = np.zeros((n_frames, height, width))
    dy_fields = np.zeros((n_frames, height, width))

    for t in range(n_frames):
        phase = 2 * np.pi * t / n_frames
        amp_x = 2.5 * np.sin(phase)
        amp_y = 1.8 * np.cos(phase)
        dx_fields[t] = amp_x * envelope
        dy_fields[t] = amp_y * envelope

    return dx_fields, dy_fields


def generate_image_sequence(ref_image, dx_fields, dy_fields, noise_sigma=0.001):
    """Forward: warp reference by displacement fields + small noise."""
    rng = np.random.RandomState(123)
    images = []
    for t in range(len(dx_fields)):
        warped = warp_image(ref_image, dx_fields[t], dy_fields[t])
        noise = rng.normal(0, noise_sigma, warped.shape)
        warped = np.clip(warped + noise, 0, 1)
        images.append(warped)
    return np.array(images)


# =====================================================================
# 3. Inverse: ZNCC (integer) + Lucas-Kanade (sub-pixel) DIC
# =====================================================================
def _interpolate_image(image, y_coords, x_coords):
    """Bilinear interpolation of image at fractional coordinates."""
    return map_coordinates(image, [y_coords, x_coords], order=1,
                           mode='reflect').reshape(y_coords.shape)


def _image_gradients(image):
    """Compute image gradients using central differences."""
    gy = np.zeros_like(image)
    gx = np.zeros_like(image)
    gy[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2.0
    gx[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2.0
    return gy, gx


def _zncc_integer_peak(ref_sub, def_region, search_margin):
    """Find integer displacement via ZNCC."""
    n_pix = ref_sub.shape[0] * ref_sub.shape[1]
    ones = np.ones_like(ref_sub)

    ref_zm = ref_sub - ref_sub.mean()
    ref_energy = np.sum(ref_zm**2)
    if ref_energy < 1e-12:
        return 0, 0

    cross = fftconvolve(def_region, ref_zm[::-1, ::-1], mode='valid')
    local_sum = fftconvolve(def_region, ones, mode='valid')
    local_sum2 = fftconvolve(def_region**2, ones, mode='valid')
    local_var = local_sum2 / n_pix - (local_sum / n_pix)**2
    local_var = np.maximum(local_var, 0.0)
    local_energy = local_var * n_pix
    denom = np.sqrt(ref_energy * local_energy)
    denom[denom < 1e-12] = 1e-12
    ncc_map = cross / denom

    peak = np.unravel_index(np.argmax(ncc_map), ncc_map.shape)
    int_dy = int(peak[0]) - search_margin
    int_dx = int(peak[1]) - search_margin
    return int_dy, int_dx


def _lucas_kanade_refine(ref_image, def_image, cy, cx, half,
                          init_dy, init_dx, n_iter=20, tol=1e-4):
    """Iterative Lucas-Kanade refinement of displacement for a single subset.

    Given an initial displacement estimate (init_dy, init_dx), iteratively
    solve for the sub-pixel correction using the gradient-based LK method.
    """
    # Reference subset coordinates
    yy, xx = np.meshgrid(
        np.arange(cy - half, cy + half, dtype=np.float64),
        np.arange(cx - half, cx + half, dtype=np.float64),
        indexing='ij'
    )
    ref_vals = ref_image[cy - half:cy + half, cx - half:cx + half].ravel()

    # Precompute gradients of deformed image
    gy_full, gx_full = _image_gradients(def_image)

    dy_curr = float(init_dy)
    dx_curr = float(init_dx)

    for iteration in range(n_iter):
        # Sample deformed image at current displacement estimate
        sample_y = yy + dy_curr
        sample_x = xx + dx_curr
        def_vals = _interpolate_image(def_image, sample_y, sample_x).ravel()

        # Gradient of deformed image at current position
        gy_vals = _interpolate_image(gy_full, sample_y, sample_x).ravel()
        gx_vals = _interpolate_image(gx_full, sample_y, sample_x).ravel()

        # Residual
        residual = ref_vals - def_vals

        # Build normal equations: [gy gx]^T [gy gx] * [ddy ddx]^T = [gy gx]^T * r
        A = np.zeros((2, 2))
        b = np.zeros(2)
        A[0, 0] = np.sum(gy_vals * gy_vals)
        A[0, 1] = np.sum(gy_vals * gx_vals)
        A[1, 0] = A[0, 1]
        A[1, 1] = np.sum(gx_vals * gx_vals)
        b[0] = np.sum(gy_vals * residual)
        b[1] = np.sum(gx_vals * residual)

        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < 1e-12:
            break

        # Solve 2x2 system
        ddy = (A[1, 1] * b[0] - A[0, 1] * b[1]) / det
        ddx = (-A[1, 0] * b[0] + A[0, 0] * b[1]) / det

        dy_curr += ddy
        dx_curr += ddx

        if abs(ddy) < tol and abs(ddx) < tol:
            break

    return dy_curr, dx_curr


def dic_zncc_lk(ref_image, def_image, subset_size=48,
                step=24, search_margin=6, lk_iterations=30):
    """
    Two-stage DIC:
      Stage 1: ZNCC for integer-pixel peak
      Stage 2: Lucas-Kanade iterative refinement for sub-pixel

    Returns dx_field, dy_field, grid_ys, grid_xs
    """
    h, w = ref_image.shape
    half = subset_size // 2
    margin = half + search_margin

    ys = np.arange(margin, h - margin, step)
    xs = np.arange(margin, w - margin, step)
    ny, nx = len(ys), len(xs)

    dx_field = np.zeros((ny, nx))
    dy_field = np.zeros((ny, nx))

    for i, cy in enumerate(ys):
        for j, cx in enumerate(xs):
            ref_sub = ref_image[cy - half:cy + half,
                                cx - half:cx + half]

            # Search region
            y0 = cy - half - search_margin
            y1 = cy + half + search_margin
            x0 = cx - half - search_margin
            x1 = cx + half + search_margin
            def_region = def_image[y0:y1, x0:x1]

            # Stage 1: integer peak
            int_dy, int_dx = _zncc_integer_peak(ref_sub, def_region,
                                                 search_margin)

            # Stage 2: Lucas-Kanade refinement
            dy_refined, dx_refined = _lucas_kanade_refine(
                ref_image, def_image, cy, cx, half,
                init_dy=float(int_dy), init_dx=float(int_dx),
                n_iter=lk_iterations
            )

            dy_field[i, j] = dy_refined
            dx_field[i, j] = dx_refined

    return dx_field, dy_field, ys, xs


# =====================================================================
# 4. Metrics
# =====================================================================
def compute_psnr(ref, test, data_range=None):
    if data_range is None:
        data_range = max(ref.max() - ref.min(), 1e-10)
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64))**2)
    if mse < 1e-30:
        return 100.0
    return float(10 * np.log10(data_range**2 / mse))


def compute_rmse(ref, test):
    return float(np.sqrt(np.mean((ref - test)**2)))


# =====================================================================
# 5. Main Pipeline
# =====================================================================
if __name__ == "__main__":
    print("=" * 64)
    print("  Task 140: pyidi_dic - DIC Displacement Tracking Pipeline")
    print("=" * 64)

    height, width = 512, 512
    n_frames = 10
    subset_size = 48
    step = 24
    search_margin = 6

    print("\n[1/5] Generating reference speckle image ...")
    ref_image = generate_speckle_image(height, width)
    print(f"       Shape: {ref_image.shape},  "
          f"range: [{ref_image.min():.3f}, {ref_image.max():.3f}]")

    print("[2/5] Generating ground-truth displacement fields ...")
    dx_gt, dy_gt = generate_displacement_fields(height, width, n_frames)
    print(f"       Frames: {n_frames}")
    print(f"       Max |dx|: {np.max(np.abs(dx_gt)):.3f} px,  "
          f"max |dy|: {np.max(np.abs(dy_gt)):.3f} px")

    print("[3/5] Generating deformed image sequence (forward) ...")
    images = generate_image_sequence(ref_image, dx_gt, dy_gt, noise_sigma=0.001)
    print(f"       Sequence shape: {images.shape}")

    print("[4/5] Running DIC (ZNCC + Lucas-Kanade sub-pixel) ...")
    all_dx_recon, all_dy_recon = [], []
    all_dx_true, all_dy_true = [], []

    for t in range(n_frames):
        dx_r, dy_r, grid_ys, grid_xs = dic_zncc_lk(
            ref_image, images[t],
            subset_size=subset_size, step=step,
            search_margin=search_margin,
            lk_iterations=30
        )
        all_dx_recon.append(dx_r)
        all_dy_recon.append(dy_r)

        dx_gt_s = dx_gt[t][np.ix_(grid_ys, grid_xs)]
        dy_gt_s = dy_gt[t][np.ix_(grid_ys, grid_xs)]
        all_dx_true.append(dx_gt_s)
        all_dy_true.append(dy_gt_s)

        errs = np.sqrt((dx_r - dx_gt_s)**2 + (dy_r - dy_gt_s)**2)
        print(f"       Frame {t:2d}: max_err={np.max(errs):.4f} px,  "
              f"mean_err={np.mean(errs):.4f} px")

    all_dx_recon = np.array(all_dx_recon)
    all_dy_recon = np.array(all_dy_recon)
    all_dx_true = np.array(all_dx_true)
    all_dy_true = np.array(all_dy_true)

    print("[5/5] Computing metrics ...")
    disp_true = np.sqrt(all_dx_true**2 + all_dy_true**2)
    disp_recon = np.sqrt(all_dx_recon**2 + all_dy_recon**2)

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
        "image_size": [height, width],
        "subset_size": subset_size,
        "step": step,
        "search_margin": search_margin,
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

    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] {metrics_path}")

    # ---- Visualisation ----
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

    print("\n" + "=" * 64)
    print("  DONE - Task 140: pyidi_dic")
    print("=" * 64)
