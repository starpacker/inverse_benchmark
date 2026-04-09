"""
Task 179: plenopticam_lf — Light field image reconstruction
Inverse problem: raw plenoptic camera data → sub-aperture images + depth estimation

Forward model:
  A scene with known depth → render through microlens array (MLA) → raw sensor image
  Disparity d(s,t) = baseline * (u - u_center) / Z(s,t)

Inverse:
  Raw MLA image → extract sub-aperture views → estimate depth via disparity
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import shift as ndi_shift, uniform_filter, gaussian_filter
from skimage.metrics import structural_similarity as ssim
import json
import os

# ─────────────────────────── Parameters ───────────────────────────
SCENE_SIZE = 128          # spatial resolution (s,t)
N_ANGULAR = 5             # angular resolution per axis (u,v) → 5×5 sub-apertures
BASELINE = 0.8            # baseline parameter controlling disparity magnitude
NOISE_STD = 0.005         # sensor noise level
PATCH_HALF = 5            # half-size of block matching window
DISP_RANGE = 6            # maximum disparity search range (pixels)
np.random.seed(42)

# ─────────────────────────── Step 1: Create scene + depth ─────────
def create_scene_and_depth(size=SCENE_SIZE):
    """Create a 2D scene with 3 depth layers and corresponding depth map."""
    scene = np.zeros((size, size), dtype=np.float64)
    depth = np.full((size, size), 5.0, dtype=np.float64)  # background depth

    # Background: smooth gradient
    yy, xx = np.mgrid[0:size, 0:size]
    scene = 0.3 + 0.3 * (xx / size) + 0.1 * np.sin(2 * np.pi * yy / size * 3)

    # Object 1: circle at depth Z=2.0 (near, large disparity)
    cy, cx, r = size // 3, size // 3, size // 6
    mask_circle = ((yy - cy) ** 2 + (xx - cx) ** 2) < r ** 2
    scene[mask_circle] = 0.85
    depth[mask_circle] = 2.0

    # Object 2: square at depth Z=3.5 (mid)
    sy, sx = int(size * 0.55), int(size * 0.55)
    half = size // 8
    mask_sq = (np.abs(yy - sy) < half) & (np.abs(xx - sx) < half)
    scene[mask_sq] = 0.55
    depth[mask_sq] = 3.5

    # Add mild texture so block matching has features to lock onto
    texture = 0.05 * np.random.RandomState(123).randn(size, size)
    texture = gaussian_filter(texture, sigma=1.0)
    scene = np.clip(scene + texture, 0, 1)

    return scene, depth


# ─────────────────────────── Step 2: Forward operator ─────────────
def forward_render_light_field(scene, depth, n_ang=N_ANGULAR, baseline=BASELINE):
    """
    Render a 4-D light field L[u, v, s, t] from a 2-D scene + depth map.

    For sub-aperture (u, v), the scene is shifted by
        dx = baseline * (u - u_c) / Z(s,t)
        dy = baseline * (v - v_c) / Z(s,t)
    """
    size = scene.shape[0]
    u_c = (n_ang - 1) / 2.0
    v_c = u_c
    lf = np.zeros((n_ang, n_ang, size, size), dtype=np.float64)

    for u in range(n_ang):
        for v in range(n_ang):
            du = baseline * (u - u_c)
            dv = baseline * (v - v_c)
            # Per-pixel disparity from depth
            disp_x = du / depth   # shape (size, size)
            disp_y = dv / depth

            # Approximate: use mean disparity per depth layer for efficiency
            shifted = np.zeros_like(scene)
            for z_val in np.unique(depth):
                mask = depth == z_val
                dx = du / z_val
                dy = dv / z_val
                layer = np.where(mask, scene, 0.0)
                shifted_layer = ndi_shift(layer, [dy, dx], order=1, mode='nearest')
                shifted += shifted_layer

            lf[u, v] = shifted

    return lf


def lf_to_raw_mla(lf):
    """
    Interleave sub-aperture images into a raw MLA sensor image.
    Raw pixel (i, j) = lf[i % n_ang, j % n_ang, i // n_ang, j // n_ang]
    """
    n_ang = lf.shape[0]
    s_size = lf.shape[2]
    raw_h = n_ang * s_size
    raw = np.zeros((raw_h, raw_h), dtype=np.float64)
    for u in range(n_ang):
        for v in range(n_ang):
            raw[u::n_ang, v::n_ang] = lf[u, v]
    return raw


def add_noise(raw, std=NOISE_STD):
    """Add Gaussian noise to the raw image."""
    noisy = raw + std * np.random.randn(*raw.shape)
    return np.clip(noisy, 0, 1)


# ─────────────────────────── Step 3: Inverse solver ───────────────
def extract_subapertures(raw, n_ang=N_ANGULAR):
    """De-interleave raw MLA image → (n_ang, n_ang, H, W) sub-apertures."""
    s_size = raw.shape[0] // n_ang
    lf = np.zeros((n_ang, n_ang, s_size, s_size), dtype=np.float64)
    for u in range(n_ang):
        for v in range(n_ang):
            lf[u, v] = raw[u::n_ang, v::n_ang]
    return lf


def estimate_depth_block_matching(lf, baseline=BASELINE,
                                  patch_half=PATCH_HALF,
                                  disp_range=DISP_RANGE):
    """
    Estimate depth by searching over candidate depth values.

    Forward model: sub-aperture (u,v) sees the scene shifted by
        dx = baseline*(u - u_c) / Z,   dy = baseline*(v - v_c) / Z

    So if we warp view (u,v) back by that shift, it should align with center.
    We discretise Z into candidates and pick the Z that gives best NCC.
    """
    n_ang = lf.shape[0]
    size = lf.shape[2]
    u_c = (n_ang - 1) / 2.0
    v_c = u_c
    center = lf[int(u_c), int(v_c)]

    # Candidate depth values — densely sample the range we know is relevant
    z_candidates = np.linspace(1.5, 6.5, 60)
    n_z = len(z_candidates)

    win = 2 * patch_half + 1
    c_mean = uniform_filter(center, size=win, mode='nearest')
    c_sq_mean = uniform_filter(center ** 2, size=win, mode='nearest')
    c_std = np.sqrt(np.maximum(c_sq_mean - c_mean ** 2, 1e-12))

    cost_vol = np.zeros((n_z, size, size), dtype=np.float64)
    count = 0

    for u in range(n_ang):
        for v in range(n_ang):
            du = u - u_c
            dv = v - v_c
            if du == 0 and dv == 0:
                continue
            count += 1
            ref = lf[u, v]
            for zi, z_val in enumerate(z_candidates):
                # The view (u,v) was created by shifting the scene by
                # (baseline*du/Z, baseline*dv/Z).
                # To undo, shift the view by the NEGATIVE of that.
                sx = -baseline * du / z_val   # column shift
                sy = -baseline * dv / z_val   # row shift
                warped = ndi_shift(ref, [sy, sx], order=1, mode='nearest')
                w_mean = uniform_filter(warped, size=win, mode='nearest')
                w_sq_mean = uniform_filter(warped ** 2, size=win, mode='nearest')
                w_std = np.sqrt(np.maximum(w_sq_mean - w_mean ** 2, 1e-12))
                cross = uniform_filter(center * warped, size=win, mode='nearest')
                ncc = (cross - c_mean * w_mean) / (c_std * w_std + 1e-12)
                cost_vol[zi] += ncc

    # Winner-take-all
    best_idx = np.argmax(cost_vol, axis=0)
    est_depth = z_candidates[best_idx]

    # Sub-pixel refinement via parabola
    for s in range(size):
        for t in range(size):
            idx = best_idx[s, t]
            if 0 < idx < n_z - 1:
                c0 = cost_vol[idx - 1, s, t]
                c1 = cost_vol[idx, s, t]
                c2 = cost_vol[idx + 1, s, t]
                denom = 2.0 * (c0 - 2 * c1 + c2)
                if abs(denom) > 1e-12:
                    offset = (c0 - c2) / denom
                    refined_idx = idx + np.clip(offset, -0.5, 0.5)
                    est_depth[s, t] = np.interp(refined_idx,
                                                np.arange(n_z), z_candidates)

    # Mild smoothing to denoise
    est_depth = gaussian_filter(est_depth, sigma=1.2)

    return est_depth, best_idx


# ─────────────────────────── Step 4: Metrics ──────────────────────
def compute_psnr(gt, est):
    mse = np.mean((gt - est) ** 2)
    if mse < 1e-15:
        return 100.0
    data_range = np.max(gt) - np.min(gt)
    if data_range < 1e-15:
        data_range = 1.0
    return 10.0 * np.log10(data_range ** 2 / mse)


def compute_cc(gt, est):
    return float(np.corrcoef(gt.ravel(), est.ravel())[0, 1])


def compute_ssim(gt, est):
    data_range = max(gt.max() - gt.min(), est.max() - est.min(), 1e-6)
    return float(ssim(gt, est, data_range=data_range))


# ─────────────────────────── Step 5: Visualization ────────────────
def visualize(gt_scene, raw_mla, recon_center, gt_depth, est_depth, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # (a) GT center sub-aperture
    ax = axes[0, 0]
    ax.imshow(gt_scene, cmap='gray', vmin=0, vmax=1)
    ax.set_title('(a) GT Center View')
    ax.axis('off')

    # (b) Raw MLA image (small patch)
    ax = axes[0, 1]
    patch_size = min(60 * N_ANGULAR, raw_mla.shape[0])
    ax.imshow(raw_mla[:patch_size, :patch_size], cmap='gray', vmin=0, vmax=1)
    ax.set_title('(b) Raw MLA Image (patch)')
    ax.axis('off')

    # (c) Reconstructed center sub-aperture
    ax = axes[0, 2]
    ax.imshow(recon_center, cmap='gray', vmin=0, vmax=1)
    ax.set_title('(c) Reconstructed Center View')
    ax.axis('off')

    # (d) GT depth map
    ax = axes[1, 0]
    vmin_d, vmax_d = gt_depth.min(), gt_depth.max()
    im_d = ax.imshow(gt_depth, cmap='viridis', vmin=vmin_d, vmax=vmax_d)
    ax.set_title('(d) GT Depth Map')
    ax.axis('off')
    plt.colorbar(im_d, ax=ax, fraction=0.046)

    # (e) Estimated depth map
    ax = axes[1, 1]
    im_e = ax.imshow(est_depth, cmap='viridis', vmin=vmin_d, vmax=vmax_d)
    ax.set_title('(e) Estimated Depth Map')
    ax.axis('off')
    plt.colorbar(im_e, ax=ax, fraction=0.046)

    # (f) Depth error map
    ax = axes[1, 2]
    err = np.abs(gt_depth - est_depth)
    im_f = ax.imshow(err, cmap='hot')
    ax.set_title('(f) Depth Error Map')
    ax.axis('off')
    plt.colorbar(im_f, ax=ax, fraction=0.046)

    plt.suptitle('Task 179: Light Field Reconstruction (plenopticam_lf)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Visualization saved to {save_path}")


# ─────────────────────────── Main pipeline ────────────────────────
def main():
    os.makedirs('results', exist_ok=True)

    # 1. Create scene + depth
    print("[1/6] Creating scene and depth map ...")
    scene, gt_depth = create_scene_and_depth()
    gt_center = scene.copy()

    # 2. Forward: render light field + raw MLA image
    print("[2/6] Forward rendering light field (5×5 views) ...")
    lf = forward_render_light_field(scene, gt_depth)
    raw_clean = lf_to_raw_mla(lf)
    raw_noisy = add_noise(raw_clean)

    # 3. Inverse: extract sub-apertures + estimate depth
    print("[3/6] Extracting sub-aperture views ...")
    lf_recon = extract_subapertures(raw_noisy)
    u_c = (N_ANGULAR - 1) // 2
    recon_center = lf_recon[u_c, u_c]

    print("[4/6] Estimating depth via block matching (this may take a moment) ...")
    est_depth, disp_map = estimate_depth_block_matching(lf_recon)

    # 4. Metrics
    print("[5/6] Computing metrics ...")
    depth_psnr = compute_psnr(gt_depth, est_depth)
    depth_cc = compute_cc(gt_depth, est_depth)
    sa_psnr = compute_psnr(gt_center, recon_center)
    sa_ssim = compute_ssim(gt_center, recon_center)

    metrics = {
        "depth_psnr_dB": round(depth_psnr, 2),
        "depth_cc": round(depth_cc, 4),
        "subaperture_psnr_dB": round(sa_psnr, 2),
        "subaperture_ssim": round(sa_ssim, 4),
        "noise_std": NOISE_STD,
        "n_angular": N_ANGULAR,
        "scene_size": SCENE_SIZE,
    }

    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  Depth  PSNR : {depth_psnr:.2f} dB")
    print(f"  Depth  CC   : {depth_cc:.4f}")
    print(f"  SA     PSNR : {sa_psnr:.2f} dB")
    print(f"  SA     SSIM : {sa_ssim:.4f}")
    print(f"{'='*50}\n")

    # 5. Visualization
    print("[6/6] Generating visualization ...")
    visualize(gt_center, raw_noisy, recon_center,
              gt_depth, est_depth, 'results/reconstruction_result.png')

    # 6. Save arrays
    np.save('results/ground_truth.npy', gt_depth)
    np.save('results/reconstruction.npy', est_depth)
    print("[DONE] All results saved to results/")


if __name__ == '__main__':
    main()
