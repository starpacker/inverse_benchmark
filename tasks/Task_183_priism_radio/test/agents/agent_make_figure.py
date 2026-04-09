import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

def make_figure(sky_gt, dirty, recon, u, v, save_path):
    """Create 5-panel figure: GT, dirty, recon, error, uv-coverage."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    vmin_log = max(sky_gt[sky_gt > 0].min() * 0.1, 1e-4) if np.any(sky_gt > 0) else 1e-4
    vmax = sky_gt.max()

    ax = axes[0, 0]
    im = ax.imshow(sky_gt, origin='lower', cmap='inferno',
                   norm=LogNorm(vmin=vmin_log, vmax=vmax))
    ax.set_title('(a) Ground Truth Sky', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Flux')

    ax = axes[0, 1]
    im = ax.imshow(dirty, origin='lower', cmap='inferno')
    ax.set_title('(b) Dirty Image', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Flux')

    ax = axes[0, 2]
    im = ax.imshow(recon, origin='lower', cmap='inferno',
                   vmin=0, vmax=vmax)
    ax.set_title('(c) L1+TSV Reconstruction', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Flux')

    ax = axes[1, 0]
    error = np.abs(sky_gt - recon)
    im = ax.imshow(error, origin='lower', cmap='hot')
    ax.set_title('(d) Error |GT - Recon|', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='|Error|')

    ax = axes[1, 1]
    ax.scatter(u, v, s=0.3, alpha=0.3, color='cyan', edgecolors='none')
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.set_xlabel('u (pixels)', fontsize=11)
    ax.set_ylabel('v (pixels)', fontsize=11)
    ax.set_title('(e) (u,v) Coverage', fontsize=13, fontweight='bold')

    ax = axes[1, 2]
    cy, cx = 64, 64
    ax.plot(sky_gt[cy, :], 'b-', linewidth=1.5, label='GT row')
    ax.plot(recon[cy, :], 'r--', linewidth=1.5, label='Recon row')
    ax.plot(sky_gt[:, cx], 'b:', linewidth=1.5, label='GT col')
    ax.plot(recon[:, cx], 'r:', linewidth=1.5, label='Recon col')
    ax.set_xlabel('Pixel', fontsize=11)
    ax.set_ylabel('Flux', fontsize=11)
    ax.set_title('(f) Cross-section (row/col through center)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

    fig.suptitle('Task 183: Radio Interferometric Imaging (L1+TSV Sparse Reconstruction)',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {save_path}")
