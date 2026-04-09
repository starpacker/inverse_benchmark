import matplotlib

matplotlib.use('Agg')

import os

import json

import numpy as np

import matplotlib.pyplot as plt

from numpy.linalg import lstsq, norm

def evaluate_results(M_true, M_recon, results_dir=None):
    """
    Compute metrics and generate visualization for Mueller matrix recovery.
    
    Parameters
    ----------
    M_true : ndarray, shape (4, 4)
        Ground truth Mueller matrix.
    M_recon : ndarray, shape (4, 4)
        Reconstructed Mueller matrix.
    results_dir : str or None
        Directory to save results. If None, uses './results'.
    
    Returns
    -------
    metrics : dict
        Contains PSNR_dB, RMSE, Frobenius_error, CC.
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Compute metrics
    diff = M_true - M_recon
    
    # Frobenius norm error
    frob_err = norm(diff, 'fro')
    
    # RMSE over all 16 elements
    rmse = np.sqrt(np.mean(diff**2))
    
    # PSNR
    max_val = np.max(np.abs(M_true))
    if max_val < 1e-15:
        max_val = 1.0
    if rmse < 1e-15:
        psnr = 100.0
    else:
        psnr = 20 * np.log10(max_val / rmse)
    
    # Element-wise Pearson correlation
    t = M_true.ravel()
    r = M_recon.ravel()
    if np.std(t) < 1e-15 or np.std(r) < 1e-15:
        cc = 1.0 if np.allclose(t, r) else 0.0
    else:
        cc = float(np.corrcoef(t, r)[0, 1])
    
    metrics = {
        'PSNR_dB': round(float(psnr), 4),
        'RMSE': round(float(rmse), 8),
        'Frobenius_error': round(float(frob_err), 8),
        'CC': round(float(cc), 6),
    }
    
    # Save arrays
    np.save(os.path.join(results_dir, 'ground_truth.npy'), M_true)
    np.save(os.path.join(results_dir, 'reconstruction.npy'), M_recon)
    
    # Save metrics
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate visualization
    fig = plt.figure(figsize=(20, 16))
    
    # Panel 1: GT Mueller matrix
    ax1 = fig.add_subplot(2, 3, 1)
    vmax = max(np.max(np.abs(M_true)), np.max(np.abs(M_recon)))
    im1 = ax1.imshow(M_true, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    ax1.set_title('Ground Truth Mueller Matrix', fontsize=12, fontweight='bold')
    for i in range(4):
        for j in range(4):
            ax1.text(j, i, f'{M_true[i,j]:.3f}', ha='center', va='center', fontsize=9,
                     color='white' if abs(M_true[i,j]) > 0.5*vmax else 'black')
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # Panel 2: Reconstructed Mueller matrix
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(M_recon, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    ax2.set_title('Reconstructed Mueller Matrix', fontsize=12, fontweight='bold')
    for i in range(4):
        for j in range(4):
            ax2.text(j, i, f'{M_recon[i,j]:.3f}', ha='center', va='center', fontsize=9,
                     color='white' if abs(M_recon[i,j]) > 0.5*vmax else 'black')
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # Panel 3: Error matrix
    ax3 = fig.add_subplot(2, 3, 3)
    err_mat = np.abs(M_true - M_recon)
    im3 = ax3.imshow(err_mat, cmap='hot', aspect='equal')
    ax3.set_title('Absolute Error |GT − Recon|', fontsize=12, fontweight='bold')
    for i in range(4):
        for j in range(4):
            ax3.text(j, i, f'{err_mat[i,j]:.4f}', ha='center', va='center', fontsize=8,
                     color='white' if err_mat[i,j] > 0.5*np.max(err_mat) else 'black')
    ax3.set_xticks(range(4))
    ax3.set_yticks(range(4))
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # Panel 4: Element-wise bar comparison
    ax4 = fig.add_subplot(2, 3, 4)
    idx = np.arange(16)
    labels = [f'M[{i},{j}]' for i in range(4) for j in range(4)]
    gt_vals = M_true.ravel()
    rc_vals = M_recon.ravel()
    width = 0.35
    ax4.bar(idx - width/2, gt_vals, width, label='Ground Truth', color='steelblue', alpha=0.8)
    ax4.bar(idx + width/2, rc_vals, width, label='Reconstructed', color='coral', alpha=0.8)
    ax4.set_xticks(idx)
    ax4.set_xticklabels(labels, rotation=65, ha='right', fontsize=7)
    ax4.set_ylabel('Element Value')
    ax4.set_title('Element-wise Comparison', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    
    # Panel 5: Scatter plot GT vs Recon
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(gt_vals, rc_vals, c='teal', s=60, edgecolors='k', linewidths=0.5, zorder=3)
    lims = [min(gt_vals.min(), rc_vals.min()) - 0.1, max(gt_vals.max(), rc_vals.max()) + 0.1]
    ax5.plot(lims, lims, 'k--', alpha=0.5, label='Ideal (y=x)')
    ax5.set_xlim(lims)
    ax5.set_ylim(lims)
    ax5.set_xlabel('Ground Truth')
    ax5.set_ylabel('Reconstructed')
    ax5.set_title(f'Correlation (CC = {metrics["CC"]:.4f})', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)
    ax5.set_aspect('equal')
    
    # Panel 6: Metrics summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    txt = (
        f"PSNR:  {metrics['PSNR_dB']:.2f} dB\n"
        f"RMSE:  {metrics['RMSE']:.6f}\n"
        f"Frobenius Error:  {metrics['Frobenius_error']:.6f}\n"
        f"Correlation (CC):  {metrics['CC']:.6f}\n"
        f"\n"
        f"Method: Dual Rotating Retarder\n"
        f"         Polarimetry (DRR)\n"
        f"Inverse: Least-Squares (pseudoinverse)\n"
        f"Sample: Partial polarizer + retarder"
    )
    ax6.text(0.1, 0.5, txt, transform=ax6.transAxes, fontsize=13,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax6.set_title('Recovery Metrics', fontsize=12, fontweight='bold')
    
    fig.suptitle('Mueller Matrix Recovery — Dual Rotating Retarder Polarimetry',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(results_dir, 'reconstruction_result.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"[SAVE] ground_truth.npy")
    print(f"[SAVE] reconstruction.npy")
    print(f"[SAVE] metrics.json")
    print(f"[SAVE] reconstruction_result.png")
    
    return metrics
