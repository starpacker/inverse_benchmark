"""
naf_cbct_recon - Sparse-View Cone-Beam CT Reconstruction
========================================================
Task: Reconstruct a 3D volume from sparse cone-beam X-ray projections.
Repo: https://github.com/Ruyi-Zha/naf_cbct

This script demonstrates the sparse-view CBCT reconstruction inverse problem:
  - Forward model: Cone-beam projection (slice-wise Radon transform)
  - Inverse: FBP + SART iterative refinement from sparse angular projections
  - Evaluation: PSNR, SSIM, RMSE

Usage: /data/yjh/naf_cbct_recon_env/bin/python naf_cbct_recon_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.transform import radon, iradon, resize
from skimage.data import shepp_logan_phantom


# ============================================================
# 1. 3D Phantom Generation
# ============================================================

def make_3d_phantom(size=64):
    """
    Generate a 3D phantom from skimage's 2D Shepp-Logan,
    with z-dependent scaling to simulate a 3D ellipsoidal volume.
    The phantom is zero outside the inscribed circle (required for
    radon(..., circle=True)).
    """
    N = size
    # Get 2D Shepp-Logan and resize
    sl2d = shepp_logan_phantom()
    sl2d = resize(sl2d, (N, N), anti_aliasing=True).astype(np.float64)

    # Ensure zero outside inscribed circle
    yy, xx = np.ogrid[-1:1:complex(N), -1:1:complex(N)]
    circle = xx ** 2 + yy ** 2 <= 1.0

    phantom = np.zeros((N, N, N), dtype=np.float64)
    for iz in range(N):
        z = 2.0 * iz / (N - 1) - 1.0
        z_env = max(1.0 - z ** 2 * 1.5, 0.0)
        if z_env < 0.01:
            continue
        # Scale the 2D phantom by z-envelope
        s = sl2d * z_env
        s[~circle] = 0.0
        phantom[iz] = s

    # Normalize to [0, 1]
    vmin, vmax = phantom.min(), phantom.max()
    if vmax > vmin:
        phantom = (phantom - vmin) / (vmax - vmin)
    phantom = np.clip(phantom, 0, 1)
    return phantom


# ============================================================
# 2. Forward Projection
# ============================================================

def forward_project(volume, angles_deg):
    """Slice-wise 2D Radon transform. Returns (D, num_det, num_angles)."""
    D = volume.shape[0]
    test = radon(volume[0], theta=angles_deg, circle=True)
    sinograms = np.zeros((D, test.shape[0], len(angles_deg)), dtype=np.float64)
    for iz in range(D):
        sinograms[iz] = radon(volume[iz], theta=angles_deg, circle=True)
    return sinograms


# ============================================================
# 3. Reconstruction
# ============================================================

def reconstruct_fbp(sinograms, angles_deg, filter_name='shepp-logan'):
    """Filtered Back-Projection reconstruction."""
    D = sinograms.shape[0]
    test = iradon(sinograms[0], theta=angles_deg, circle=True)
    N = test.shape[0]
    recon = np.zeros((D, N, N), dtype=np.float64)
    for iz in range(D):
        r = iradon(sinograms[iz], theta=angles_deg, circle=True,
                   filter_name=filter_name)
        recon[iz] = r
    return recon


def reconstruct_sart(sinograms, angles_deg, num_iterations=15, relaxation=0.02):
    """
    SART-like iterative reconstruction with FBP initialization.
    Refines FBP result by iteratively minimizing the sinogram residual.
    """
    D = sinograms.shape[0]
    test = iradon(sinograms[0], theta=angles_deg, circle=True)
    N = test.shape[0]
    recon = np.zeros((D, N, N), dtype=np.float64)

    # Normalization volume: back-project a uniform sinogram
    ones_sino = np.ones_like(sinograms[0])
    norm_vol = iradon(ones_sino, theta=angles_deg, circle=True, filter_name=None)
    norm_vol = np.clip(np.abs(norm_vol), 1e-6, None)

    for iz in range(D):
        sino_meas = sinograms[iz]
        # FBP init
        r = iradon(sino_meas, theta=angles_deg, circle=True)
        r = np.clip(r, 0, None)

        for it in range(num_iterations):
            sino_est = radon(r, theta=angles_deg, circle=True)
            residual = sino_meas - sino_est
            bp_res = iradon(residual, theta=angles_deg, circle=True,
                            filter_name=None)
            r = r + relaxation * bp_res / norm_vol
            r = np.clip(r, 0, None)

        recon[iz] = r

    return recon


# ============================================================
# 4. Metrics
# ============================================================

def compute_metrics(gt, recon):
    """Compute PSNR, SSIM, RMSE on [0,1]-normalized data."""
    gt_f = gt.astype(np.float64)
    recon_f = recon.astype(np.float64)

    vmin, vmax = gt_f.min(), gt_f.max()
    if vmax - vmin > 1e-10:
        gt_n = (gt_f - vmin) / (vmax - vmin)
        recon_n = np.clip((recon_f - vmin) / (vmax - vmin), 0, 1)
    else:
        gt_n, recon_n = gt_f, recon_f

    rmse_val = np.sqrt(np.mean((gt_n - recon_n) ** 2))
    psnr_list, ssim_list = [], []
    for iz in range(gt_n.shape[0]):
        gs, rs = gt_n[iz], recon_n[iz]
        if gs.max() - gs.min() < 1e-8:
            continue
        psnr_list.append(psnr(gs, rs, data_range=1.0))
        ssim_list.append(ssim(gs, rs, data_range=1.0))

    return {
        'psnr': round(np.mean(psnr_list) if psnr_list else 0.0, 4),
        'ssim': round(np.mean(ssim_list) if ssim_list else 0.0, 4),
        'rmse': round(float(rmse_val), 6)
    }


# ============================================================
# 5. Visualization
# ============================================================

def visualize(gt, recon_s, recon_f, ms, mf, sino_f, sino_s, af, a_s, path):
    D, H, W = gt.shape
    md, mh, mw = D // 2, H // 2, W // 2

    vmin, vmax = gt.min(), gt.max()
    norm = lambda x: np.clip((x - vmin) / (vmax - vmin), 0, 1) if vmax > vmin else x

    gd, rfd, rsd = norm(gt), norm(recon_f), norm(recon_s)

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.suptitle(
        'Sparse-View Cone-Beam CT Reconstruction\n'
        f'Sparse ({len(a_s)} views): PSNR={ms["psnr"]:.2f} dB, SSIM={ms["ssim"]:.4f}  |  '
        f'Full ({len(af)} views): PSNR={mf["psnr"]:.2f} dB, SSIM={mf["ssim"]:.4f}',
        fontsize=14, fontweight='bold')

    views = [('Axial', lambda x: x[md], 'equal'),
             ('Coronal', lambda x: x[:, mh, :], 'auto'),
             ('Sagittal', lambda x: x[:, :, mw], 'auto')]

    for row, (name, sl, asp) in enumerate(views):
        axes[row, 0].imshow(sl(gd), cmap='gray', vmin=0, vmax=1, aspect=asp)
        axes[row, 0].set_title(f'GT ({name})'); axes[row, 0].axis('off')

        axes[row, 1].imshow(sl(rfd), cmap='gray', vmin=0, vmax=1, aspect=asp)
        axes[row, 1].set_title(f'Full Recon ({name})'); axes[row, 1].axis('off')

        axes[row, 2].imshow(sl(rsd), cmap='gray', vmin=0, vmax=1, aspect=asp)
        axes[row, 2].set_title(f'Sparse Recon ({name})'); axes[row, 2].axis('off')

        err = np.abs(sl(gd) - sl(rsd))
        im = axes[row, 3].imshow(err, cmap='hot', vmin=0, vmax=0.3, aspect=asp)
        axes[row, 3].set_title(f'Error ({name})'); axes[row, 3].axis('off')
        plt.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)

    # Sinograms
    axes[3, 0].imshow(sino_f[md], cmap='gray', aspect='auto')
    axes[3, 0].set_title(f'Full Sinogram ({len(af)} ang)'); axes[3, 0].set_xlabel('Angle')

    axes[3, 1].imshow(sino_s[md], cmap='gray', aspect='auto')
    axes[3, 1].set_title(f'Sparse Sinogram ({len(a_s)} ang)'); axes[3, 1].set_xlabel('Angle')

    axes[3, 2].plot(gd[md, mh], 'k-', lw=2, label='GT')
    axes[3, 2].plot(rfd[md, mh], 'b--', lw=1.5, label=f'Full ({len(af)})')
    axes[3, 2].plot(rsd[md, mh], 'r-.', lw=1.5, label=f'Sparse ({len(a_s)})')
    axes[3, 2].legend(fontsize=8); axes[3, 2].set_title('Line Profile'); axes[3, 2].grid(alpha=0.3)

    axes[3, 3].axis('off')
    txt = (f"Volume: {D}x{H}x{W}\n\n"
           f"Full ({len(af)} ang):\n  PSNR={mf['psnr']:.2f}dB\n  SSIM={mf['ssim']:.4f}\n  RMSE={mf['rmse']:.6f}\n\n"
           f"Sparse ({len(a_s)} ang):\n  PSNR={ms['psnr']:.2f}dB\n  SSIM={ms['ssim']:.4f}\n  RMSE={ms['rmse']:.6f}\n\n"
           f"Method: FBP + SART")
    axes[3, 3].text(0.1, 0.95, txt, transform=axes[3, 3].transAxes, fontsize=11, va='top',
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Sparse-View Cone-Beam CT Reconstruction (Task 194)")
    print("=" * 60)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(out_dir, exist_ok=True)

    vol_size = 64
    n_full, n_sparse = 180, 50
    sart_iters, sart_relax = 10, 0.02

    # 1. Phantom
    print("\n[1/5] Generating 3D phantom...")
    t0 = time.time()
    phantom = make_3d_phantom(vol_size)
    print(f"  Shape: {phantom.shape}, range: [{phantom.min():.4f}, {phantom.max():.4f}], t={time.time()-t0:.1f}s")

    # 2. Forward projection
    af = np.linspace(0, 180, n_full, endpoint=False)
    a_s = np.linspace(0, 180, n_sparse, endpoint=False)
    print(f"\n[2/5] Forward projection...")
    t0 = time.time()
    sino_f = forward_project(phantom, af)
    sino_s = forward_project(phantom, a_s)
    print(f"  Full: {sino_f.shape}, Sparse: {sino_s.shape}, t={time.time()-t0:.1f}s")

    # 3. Full recon
    print(f"\n[3/5] Full-view SART ({n_full} angles)...")
    t0 = time.time()
    recon_f = reconstruct_sart(sino_f, af, sart_iters, sart_relax)
    print(f"  range: [{recon_f.min():.4f}, {recon_f.max():.4f}], t={time.time()-t0:.1f}s")

    # 4. Sparse recon
    print(f"\n[4/5] Sparse-view SART ({n_sparse} angles)...")
    t0 = time.time()
    recon_s = reconstruct_sart(sino_s, a_s, sart_iters, sart_relax)
    print(f"  range: [{recon_s.min():.4f}, {recon_s.max():.4f}], t={time.time()-t0:.1f}s")

    # 5. Metrics
    print("\n[5/5] Metrics...")
    mf = compute_metrics(phantom, recon_f)
    ms = compute_metrics(phantom, recon_s)
    print(f"  Full:   PSNR={mf['psnr']:.2f}dB, SSIM={mf['ssim']:.4f}, RMSE={mf['rmse']:.6f}")
    print(f"  Sparse: PSNR={ms['psnr']:.2f}dB, SSIM={ms['ssim']:.4f}, RMSE={ms['rmse']:.6f}")

    # Save
    out = {
        'task': 'naf_cbct_recon', 'task_id': 194, 'method': 'FBP_SART',
        'volume_size': vol_size, 'num_angles_full': n_full, 'num_angles_sparse': n_sparse,
        'sart_iterations': sart_iters,
        'psnr': ms['psnr'], 'ssim': ms['ssim'], 'rmse': ms['rmse'],
        'full_view_psnr': mf['psnr'], 'full_view_ssim': mf['ssim'], 'full_view_rmse': mf['rmse'],
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(out, f, indent=2)

    np.save(os.path.join(out_dir, 'ground_truth.npy'), phantom.astype(np.float32))
    np.save(os.path.join(out_dir, 'reconstruction.npy'), recon_s.astype(np.float32))
    np.save(os.path.join(out_dir, 'reconstruction_full.npy'), recon_f.astype(np.float32))

    visualize(phantom, recon_s, recon_f, ms, mf, sino_f, sino_s, af, a_s,
              os.path.join(out_dir, 'reconstruction_result.png'))

    print(f"\n{'='*60}")
    print(f"DONE. Sparse PSNR={ms['psnr']:.2f}dB, SSIM={ms['ssim']:.4f}")
    print(f"{'='*60}")
    return out


if __name__ == '__main__':
    main()
