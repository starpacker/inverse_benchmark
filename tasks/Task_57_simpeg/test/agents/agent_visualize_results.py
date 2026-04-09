import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def visualize_results(mesh_info, model_gt, model_rec, rx_locs,
                      d_clean, d_noisy, d_rec, metrics, save_path):
    """Create visualization of inversion results."""
    nx, ny, nz = mesh_info['shape_cells']
    gt_3d = model_gt.reshape((nx, ny, nz), order='F')
    rec_3d = model_rec.reshape((nx, ny, nz), order='F')

    iz = nz // 2

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    vmax = max(np.abs(gt_3d).max(), 0.1)

    # (a) GT slice
    im = axes[0, 0].imshow(gt_3d[:, :, iz].T, cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax, origin='lower')
    axes[0, 0].set_title(f'(a) GT Density (z-slice {iz})')
    plt.colorbar(im, ax=axes[0, 0], label='Δρ [g/cm³]')

    # (b) Reconstructed slice
    im = axes[0, 1].imshow(rec_3d[:, :, iz].T, cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax, origin='lower')
    axes[0, 1].set_title('(b) Inversion Result')
    plt.colorbar(im, ax=axes[0, 1], label='Δρ [g/cm³]')

    # (c) Error
    err = gt_3d[:, :, iz] - rec_3d[:, :, iz]
    im = axes[0, 2].imshow(err.T, cmap='RdBu_r',
                            vmin=-vmax/2, vmax=vmax/2, origin='lower')
    axes[0, 2].set_title('(c) Error')
    plt.colorbar(im, ax=axes[0, 2], label='Δρ error')

    # (d) Observed data map
    n_rx = int(np.sqrt(len(d_clean)))
    if n_rx ** 2 == len(d_clean):
        d_map = d_clean.reshape(n_rx, n_rx)
        axes[1, 0].imshow(d_map, cmap='viridis', origin='lower')
    else:
        axes[1, 0].scatter(rx_locs[:, 0], rx_locs[:, 1],
                           c=d_clean, cmap='viridis', s=20)
    axes[1, 0].set_title('(d) Gravity Anomaly (GT)')

    # (e) Data fit
    axes[1, 1].plot(d_clean, d_rec, 'b.', ms=3)
    lims = [min(d_clean.min(), d_rec.min()),
            max(d_clean.max(), d_rec.max())]
    axes[1, 1].plot(lims, lims, 'k--', lw=0.5)
    axes[1, 1].set_xlabel('True g_z [mGal]')
    axes[1, 1].set_ylabel('Predicted g_z [mGal]')
    axes[1, 1].set_title(f'(e) Data Fit  CC={metrics["CC_data"]:.4f}')

    # (f) Depth profile
    axes[1, 2].plot(gt_3d[nx//2, ny//2, :], range(nz), 'b-', lw=2, label='GT')
    axes[1, 2].plot(rec_3d[nx//2, ny//2, :], range(nz), 'r--', lw=2, label='Inv')
    axes[1, 2].set_xlabel('Δρ [g/cm³]')
    axes[1, 2].set_ylabel('Depth index')
    axes[1, 2].set_title('(f) Depth Profile')
    axes[1, 2].legend()
    axes[1, 2].invert_yaxis()

    fig.suptitle(
        f"Gravity Anomaly Inversion\n"
        f"PSNR={metrics['PSNR_slice']:.1f} dB  |  "
        f"SSIM={metrics['SSIM_slice']:.4f}  |  "
        f"CC_vol={metrics['CC_volume']:.4f}",
        fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")
