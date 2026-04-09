import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim_fn

from scipy.ndimage import label

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_results(spec_gt, spec_recon, fid_nus, schedule, n_f1, n_peaks, nus_frac, results_dir):
    """
    Compute metrics and generate visualization.
    
    Parameters
    ----------
    spec_gt : np.ndarray
        Ground truth spectrum.
    spec_recon : np.ndarray
        Reconstructed spectrum.
    fid_nus : np.ndarray
        NUS-sampled FID.
    schedule : np.ndarray
        Boolean NUS sampling mask.
    n_f1 : int
        Number of points in indirect dimension.
    n_peaks : int
        Number of peaks in ground truth.
    nus_frac : float
        NUS sampling fraction.
    results_dir : str
        Directory to save results.
    
    Returns
    -------
    dict
        Metrics dictionary.
    """
    print("\n[EVAL] Computing metrics ...")
    
    # Normalise both
    gt = spec_gt / np.abs(spec_gt).max()
    rec = spec_recon / np.abs(spec_recon).max()

    # PSNR
    data_range = gt.max() - gt.min()
    mse = np.mean((gt - rec) ** 2)
    psnr = float(10 * np.log10(data_range ** 2 / max(mse, 1e-30)))

    # SSIM
    ssim_val = float(ssim_fn(gt, rec, data_range=data_range))

    # CC
    cc = float(np.corrcoef(gt.ravel(), rec.ravel())[0, 1])

    # Relative error
    re = float(np.linalg.norm(gt - rec) / max(np.linalg.norm(gt), 1e-12))

    # RMSE
    rmse = float(np.sqrt(mse))

    # Peak detection accuracy
    gt_mask = gt > 0.15 * gt.max()
    rec_mask = rec > 0.15 * rec.max()
    gt_labels, n_gt = label(gt_mask)
    rec_labels, n_rec = label(rec_mask)

    metrics = {
        "PSNR": psnr,
        "SSIM": ssim_val,
        "CC": cc,
        "RE": re,
        "RMSE": rmse,
        "n_peaks_gt": int(n_gt),
        "n_peaks_recon": int(n_rec),
    }

    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")

    # Save metrics and data
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(results_dir, "reconstruction.npy"), spec_recon)
    np.save(os.path.join(results_dir, "ground_truth.npy"), spec_gt)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    vmax = np.percentile(np.abs(spec_gt), 99)

    # (a) Ground truth spectrum
    ax = axes[0, 0]
    ax.contourf(spec_gt.T, levels=30, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f'(a) Ground Truth ({n_peaks} peaks)')
    ax.set_xlabel('F1 [pts]')
    ax.set_ylabel('F2 [pts]')

    # (b) Reconstructed spectrum
    ax = axes[0, 1]
    ax.contourf(spec_recon.T, levels=30, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f'(b) IST Reconstruction (NUS {nus_frac*100:.0f}%)')
    ax.set_xlabel('F1 [pts]')
    ax.set_ylabel('F2 [pts]')

    # (c) NUS schedule
    ax = axes[1, 0]
    ax.stem(np.where(schedule)[0], np.ones(schedule.sum()),
            linefmt='b-', markerfmt='b.', basefmt='k-')
    ax.set_xlim(0, n_f1)
    ax.set_xlabel('Indirect dimension index')
    ax.set_ylabel('Sampled')
    ax.set_title(f'(c) NUS Schedule ({schedule.sum()}/{n_f1})')

    # (d) 1D slice comparison
    ax = axes[1, 1]
    mid = spec_gt.shape[1] // 2
    ax.plot(spec_gt[:, mid], 'b-', lw=1.5, label='GT', alpha=0.8)
    ax.plot(spec_recon[:, mid], 'r--', lw=1.5, label='IST recon', alpha=0.8)
    ax.set_xlabel('F1 [pts]')
    ax.set_ylabel('Intensity')
    ax.set_title('(d) 1D Slice (F2 midpoint)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"nmrglue — 2D NMR NUS Reconstruction (IST)\n"
        f"PSNR={metrics['PSNR']:.1f} dB  |  SSIM={metrics['SSIM']:.4f}  |  "
        f"CC={metrics['CC']:.4f}  |  RE={metrics['RE']:.4f}",
        fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")

    return metrics
