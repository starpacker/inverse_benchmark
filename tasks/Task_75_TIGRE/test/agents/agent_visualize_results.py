import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def visualize_results(gt, sinogram, rec_fbp, rec_sirt, angles, metrics, save_path):
    """
    Create visualization of reconstruction results.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].imshow(gt, cmap='gray')
    axes[0, 0].set_title('Ground Truth (Shepp-Logan)')

    axes[0, 1].imshow(sinogram, aspect='auto', cmap='gray')
    axes[0, 1].set_title(f'Sinogram ({len(angles)} angles)')
    axes[0, 1].set_xlabel('Detector')
    axes[0, 1].set_ylabel('Angle index')

    axes[0, 2].imshow(rec_fbp / max(rec_fbp.max(), 1e-12), cmap='gray')
    axes[0, 2].set_title('FBP Reconstruction')

    axes[1, 0].imshow(rec_sirt / max(rec_sirt.max(), 1e-12), cmap='gray')
    axes[1, 0].set_title('SIRT Reconstruction')

    err = np.abs(gt / max(gt.max(), 1e-12) - rec_sirt / max(rec_sirt.max(), 1e-12))
    axes[1, 1].imshow(err, cmap='hot')
    axes[1, 1].set_title('|Error| (SIRT)')

    # Profile comparison
    mid = gt.shape[0] // 2
    axes[1, 2].plot(gt[mid, :] / max(gt[mid, :].max(), 1e-12), 'b-', lw=2, label='GT')
    axes[1, 2].plot(rec_fbp[mid, :] / max(rec_fbp[mid, :].max(), 1e-12),
                     'g--', lw=1.5, label='FBP')
    axes[1, 2].plot(rec_sirt[mid, :] / max(rec_sirt[mid, :].max(), 1e-12),
                     'r--', lw=1.5, label='SIRT')
    axes[1, 2].set_title('Central Profile')
    axes[1, 2].legend()

    n_angles_sparse = len(angles)
    fig.suptitle(
        f"TIGRE — Sparse-View CT Reconstruction ({n_angles_sparse} views)\n"
        f"PSNR={metrics['PSNR']:.1f} dB | SSIM={metrics['SSIM']:.4f} | "
        f"CC={metrics['CC']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
