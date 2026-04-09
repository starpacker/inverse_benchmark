import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

def _visualize_results(gt_3d, recon_3d, n_views, reproj_errors,
                       cameras, metrics, config, save_path):
    """
    Create comprehensive visualization.
    """
    vol_min = config['vol_min']
    vol_max = config['vol_max']
    n_cameras = config['n_cameras']
    
    valid = ~np.isnan(recon_3d[:, 0])
    gt_v = gt_3d[valid]
    rc_v = recon_3d[valid]
    errors_3d = np.linalg.norm(gt_v - rc_v, axis=1)

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(
        "3D Particle Tracking Velocimetry — Multi-Camera Triangulation\n"
        f"RMSE={metrics['rmse_3d_mm']:.3f} mm  |  "
        f"CC={metrics['correlation_mean']:.4f}  |  "
        f"PSNR={metrics['psnr_db']:.1f} dB  |  "
        f"Success={metrics['success_rate']*100:.1f}%",
        fontsize=14, fontweight='bold'
    )

    # (a) 3D scatter: GT (blue) vs Recon (red)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(gt_v[:, 0], gt_v[:, 1], gt_v[:, 2],
                c='blue', s=8, alpha=0.5, label='Ground Truth')
    ax1.scatter(rc_v[:, 0], rc_v[:, 1], rc_v[:, 2],
                c='red', s=8, alpha=0.5, label='Reconstructed')
    for cam in cameras:
        cp = cam['cam_pos']
        ax1.scatter(*cp, c='green', s=100, marker='^', zorder=5)
    ax1.set_xlabel('X [mm]')
    ax1.set_ylabel('Y [mm]')
    ax1.set_zlabel('Z [mm]')
    ax1.set_title('3D Positions: GT vs Recon')
    ax1.legend(fontsize=8)

    # (b) X-axis correlation
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(gt_v[:, 0], rc_v[:, 0], s=10, alpha=0.6, c='steelblue')
    lim = [vol_min[0] - 5, vol_max[0] + 5]
    ax2.plot(lim, lim, 'k--', lw=1)
    ax2.set_xlabel('GT X [mm]')
    ax2.set_ylabel('Recon X [mm]')
    ax2.set_title(f'X Correlation (CC={metrics["correlation_x"]:.4f})')
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # (c) Y-axis correlation
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(gt_v[:, 1], rc_v[:, 1], s=10, alpha=0.6, c='coral')
    lim_y = [vol_min[1] - 5, vol_max[1] + 5]
    ax3.plot(lim_y, lim_y, 'k--', lw=1)
    ax3.set_xlabel('GT Y [mm]')
    ax3.set_ylabel('Recon Y [mm]')
    ax3.set_title(f'Y Correlation (CC={metrics["correlation_y"]:.4f})')
    ax3.set_xlim(lim_y)
    ax3.set_ylim(lim_y)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # (d) Z-axis correlation
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(gt_v[:, 2], rc_v[:, 2], s=10, alpha=0.6, c='green')
    lim_z = [vol_min[2] - 5, vol_max[2] + 5]
    ax4.plot(lim_z, lim_z, 'k--', lw=1)
    ax4.set_xlabel('GT Z [mm]')
    ax4.set_ylabel('Recon Z [mm]')
    ax4.set_title(f'Z Correlation (CC={metrics["correlation_z"]:.4f})')
    ax4.set_xlim(lim_z)
    ax4.set_ylim(lim_z)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)

    # (e) 3D error histogram
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(errors_3d, bins=40, color='steelblue', edgecolor='black',
             alpha=0.7)
    ax5.axvline(metrics['rmse_3d_mm'], color='red', ls='--',
                label=f'RMSE={metrics["rmse_3d_mm"]:.3f} mm')
    ax5.axvline(metrics['median_error_mm'], color='orange', ls='--',
                label=f'Median={metrics["median_error_mm"]:.3f} mm')
    ax5.set_xlabel('3D Position Error [mm]')
    ax5.set_ylabel('Count')
    ax5.set_title('Error Distribution')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # (f) Number of views histogram
    ax6 = fig.add_subplot(2, 3, 6)
    view_counts = n_views[valid]
    bins_v = np.arange(0.5, n_cameras + 1.5, 1)
    ax6.hist(view_counts, bins=bins_v, color='mediumpurple',
             edgecolor='black', alpha=0.7)
    ax6.set_xlabel('Number of Camera Views')
    ax6.set_ylabel('Count')
    ax6.set_title('Views per Particle')
    ax6.set_xticks(range(1, n_cameras + 1))
    ax6.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")
