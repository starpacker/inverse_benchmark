import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os

import sys

import json

import torch

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")

if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_psnr(ref, test, data_range=None):
    """Compute PSNR (dB)."""
    if data_range is None:
        data_range = ref.max() - ref.min()
    mse = np.mean((ref.astype(float) - test.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(data_range ** 2 / mse)

def compute_ssim(ref, test):
    """Compute SSIM."""
    from skimage.metrics import structural_similarity as ssim
    data_range = ref.max() - ref.min()
    if data_range == 0:
        data_range = 1.0
    return ssim(ref, test, data_range=data_range)

def compute_rmse(ref, test):
    """Compute RMSE."""
    return np.sqrt(np.mean((ref.astype(float) - test.astype(float)) ** 2))

def compute_relative_l2(ref, test):
    """Compute relative L2 error."""
    return np.linalg.norm(ref - test) / (np.linalg.norm(ref) + 1e-10)

def evaluate_results(model, data_dict, device, results_dir):
    """
    Evaluate the trained FNO model on the test set and generate visualizations.
    
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\n[EVAL] Evaluating on test set...")
    
    a_test = data_dict['a_test']
    u_test = data_dict['u_test']
    test_loader = data_dict['test_loader']
    n_test = len(u_test)
    
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for a_batch, u_batch in test_loader:
            a_batch = a_batch.to(device)
            pred = model(a_batch)
            all_preds.append(pred.cpu().numpy())
    
    predictions = np.concatenate(all_preds, axis=0)
    
    all_psnr, all_ssim, all_rmse, all_rel_l2 = [], [], [], []
    
    for i in range(n_test):
        gt_i = u_test[i]
        pred_i = predictions[i, 0]
        
        all_psnr.append(compute_psnr(gt_i, pred_i))
        all_ssim.append(compute_ssim(gt_i, pred_i))
        all_rmse.append(compute_rmse(gt_i, pred_i))
        all_rel_l2.append(compute_relative_l2(gt_i, pred_i))
    
    metrics = {
        "psnr": float(np.mean(all_psnr)),
        "ssim": float(np.mean(all_ssim)),
        "rmse": float(np.mean(all_rmse)),
        "relative_l2": float(np.mean(all_rel_l2)),
        "psnr_std": float(np.std(all_psnr)),
        "ssim_std": float(np.std(all_ssim)),
        "n_test": n_test,
    }
    
    print(f"[EVAL] Mean PSNR = {metrics['psnr']:.4f} dB (±{metrics['psnr_std']:.2f})")
    print(f"[EVAL] Mean SSIM = {metrics['ssim']:.6f} (±{metrics['ssim_std']:.4f})")
    print(f"[EVAL] Mean RMSE = {metrics['rmse']:.6f}")
    print(f"[EVAL] Mean Rel. L2 = {metrics['relative_l2']:.6f}")
    
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics → {metrics_path}")
    
    best_idx = int(np.argmax(all_psnr))
    sample_metrics = {
        "psnr": all_psnr[best_idx],
        "ssim": all_ssim[best_idx],
        "rmse": all_rmse[best_idx],
        "relative_l2": all_rel_l2[best_idx],
    }
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    im0 = axes[0].imshow(a_test[best_idx], cmap='viridis', origin='lower')
    axes[0].set_title('Input: Permeability a(x)', fontsize=12)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    im1 = axes[1].imshow(u_test[best_idx], cmap='RdBu_r', origin='lower')
    axes[1].set_title('GT: PDE Solution u(x)', fontsize=12)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    im2 = axes[2].imshow(predictions[best_idx, 0], cmap='RdBu_r', origin='lower',
                         vmin=u_test[best_idx].min(), vmax=u_test[best_idx].max())
    axes[2].set_title('FNO Prediction', fontsize=12)
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    error = np.abs(u_test[best_idx] - predictions[best_idx, 0])
    im3 = axes[3].imshow(error, cmap='hot', origin='lower')
    axes[3].set_title('|Error|', fontsize=12)
    plt.colorbar(im3, ax=axes[3], fraction=0.046)
    
    fig.suptitle(
        f"FNO Darcy Flow | PSNR={sample_metrics['psnr']:.2f} dB | SSIM={sample_metrics['ssim']:.4f} | "
        f"Rel. L2={sample_metrics['relative_l2']:.4f}",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {vis_path}")
    
    np.save(os.path.join(results_dir, "reconstruction.npy"), predictions[best_idx, 0])
    np.save(os.path.join(results_dir, "ground_truth.npy"), u_test[best_idx])
    np.save(os.path.join(results_dir, "input_coefficient.npy"), a_test[best_idx])
    
    return metrics
