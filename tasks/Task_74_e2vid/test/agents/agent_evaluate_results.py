import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import json

from skimage.metrics import structural_similarity as ssim_fn

def evaluate_results(gt_frames, rec_frames, events, results_dir, working_dir):
    """
    Compute metrics and save results.
    
    Args:
        gt_frames: Ground truth frames
        rec_frames: Reconstructed frames
        events: List of events
        results_dir: Directory to save results
        working_dir: Working directory
    
    Returns:
        metrics: Dictionary of computed metrics
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Compute metrics
    n = min(len(gt_frames), len(rec_frames))
    psnr_list, ssim_list, cc_list = [], [], []

    for i in range(n):
        gt = gt_frames[i]
        rec = rec_frames[i]
        # Min-max normalization to [0, 1]
        gt_n = (gt - gt.min()) / (gt.max() - gt.min() + 1e-10)
        rec_n = (rec - rec.min()) / (rec.max() - rec.min() + 1e-10)

        data_range = 1.0
        mse = np.mean((gt_n - rec_n)**2)
        psnr_list.append(10 * np.log10(data_range**2 / max(mse, 1e-30)))
        ssim_list.append(ssim_fn(gt_n, rec_n, data_range=data_range))
        cc_list.append(np.corrcoef(gt_n.ravel(), rec_n.ravel())[0, 1])

    gt_all = gt_frames[:n].ravel()
    rec_all = rec_frames[:n].ravel()
    gt_all_n = (gt_all - gt_all.min()) / (gt_all.max() - gt_all.min() + 1e-10)
    rec_all_n = (rec_all - rec_all.min()) / (rec_all.max() - rec_all.min() + 1e-10)
    re = float(np.linalg.norm(gt_all_n - rec_all_n) /
               max(np.linalg.norm(gt_all_n), 1e-12))
    rmse = float(np.sqrt(np.mean((gt_all_n - rec_all_n)**2)))

    metrics = {
        "PSNR": float(np.mean(psnr_list)),
        "SSIM": float(np.mean(ssim_list)),
        "CC": float(np.mean(cc_list)),
        "RE": re,
        "RMSE": rmse,
    }
    
    # Save metrics
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save arrays
    np.save(os.path.join(results_dir, "reconstruction.npy"), rec_frames)
    np.save(os.path.join(results_dir, "ground_truth.npy"), gt_frames)
    np.save(os.path.join(working_dir, "recon_output.npy"), rec_frames)
    np.save(os.path.join(working_dir, "gt_output.npy"), gt_frames)
    
    # Visualization
    n_show = min(4, len(gt_frames))
    fig, axes = plt.subplots(3, n_show, figsize=(4 * n_show, 10))

    indices = np.linspace(0, len(gt_frames) - 1, n_show, dtype=int)

    for col, idx in enumerate(indices):
        gt = gt_frames[idx] / max(gt_frames[idx].max(), 1e-12)
        rec = rec_frames[idx] / max(rec_frames[idx].max(), 1e-12)

        axes[0, col].imshow(gt, cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(f'GT Frame {idx}')
        axes[0, col].axis('off')

        axes[1, col].imshow(rec, cmap='gray', vmin=0, vmax=1)
        axes[1, col].set_title(f'Recon Frame {idx}')
        axes[1, col].axis('off')

        axes[2, col].imshow(np.abs(gt - rec), cmap='hot', vmin=0, vmax=0.5)
        axes[2, col].set_title('|Error|')
        axes[2, col].axis('off')

    fig.suptitle(
        f"e2vid — Event Camera Reconstruction\n"
        f"Events: {len(events)} | PSNR={metrics['PSNR']:.1f} dB | "
        f"SSIM={metrics['SSIM']:.4f} | CC={metrics['CC']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(results_dir, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics
