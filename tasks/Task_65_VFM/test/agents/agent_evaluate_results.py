import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim_fn

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_results(data, result, save_dir):
    """
    Compute metrics, save results, and generate visualizations.
    
    Parameters:
        data: Dictionary containing ground truth data
        result: Dictionary containing reconstruction results
        save_dir: Directory to save results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    sig_gt = data['sigma_xx_gt']
    sig_rec = result['sigma_xx_rec']
    E_gt = data['gt_E']
    E_rec = result['E_rec']
    nu_gt = data['gt_nu']
    nu_rec = result['nu_rec']
    xx = data['xx']
    yy = data['yy']
    
    # Compute metrics
    dr = sig_gt.max() - sig_gt.min()
    if dr < 1e-12:
        dr = 1.0
    mse = np.mean((sig_gt - sig_rec)**2)
    psnr = float(10*np.log10(dr**2/max(mse, 1e-30)))
    ssim_val = float(ssim_fn(sig_gt, sig_rec, data_range=dr))
    cc = float(np.corrcoef(sig_gt.ravel(), sig_rec.ravel())[0, 1])
    re = float(np.linalg.norm(sig_gt - sig_rec)/max(np.linalg.norm(sig_gt), 1e-12))
    
    metrics = {
        "PSNR": psnr,
        "SSIM": ssim_val,
        "CC": cc,
        "RE": re,
        "E_gt": float(E_gt),
        "E_rec": float(E_rec),
        "E_err_pct": float(abs(E_gt - E_rec)/E_gt*100),
        "nu_gt": float(nu_gt),
        "nu_rec": float(nu_rec),
        "nu_err_pct": float(abs(nu_gt - nu_rec)/nu_gt*100)
    }
    
    # Print metrics
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")
    
    # Save metrics to JSON
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save reconstructions
    np.save(os.path.join(save_dir, "reconstruction.npy"), sig_rec)
    np.save(os.path.join(save_dir, "ground_truth.npy"), sig_gt)
    
    # Generate visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmax = max(np.abs(sig_gt).max(), np.abs(sig_rec).max())
    
    for ax, arr, title in zip(axes[:2], [sig_gt, sig_rec],
                               ['GT σ_xx', 'VFM σ_xx']):
        im = ax.imshow(arr.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       origin='lower', aspect='auto')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    
    err = sig_gt - sig_rec
    im = axes[2].imshow(err.T, cmap='RdBu_r', origin='lower', aspect='auto')
    axes[2].set_title('Error')
    plt.colorbar(im, ax=axes[2])
    
    fig.suptitle(f"VFM — E={metrics['E_rec']:.0f} MPa (err {metrics['E_err_pct']:.1f}%)  |  "
                 f"ν={metrics['nu_rec']:.3f} (err {metrics['nu_err_pct']:.1f}%)  |  "
                 f"CC={metrics['CC']:.4f}", fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    save_path = os.path.join(save_dir, "reconstruction_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")
    
    return metrics
