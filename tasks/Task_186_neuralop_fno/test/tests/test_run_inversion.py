import sys
import os
import dill
import numpy as np
import traceback
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage.metrics import structural_similarity as ssim

# Add repo to path
REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

from agent_run_inversion import run_inversion

# ============================================================
# Referee: evaluate_results (injected verbatim from Reference B)
# ============================================================

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
    """
    print("\n[EVAL] Evaluating on test set...")

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

    print("[EVAL] Mean PSNR = {:.4f} dB (+/-{:.2f})".format(metrics['psnr'], metrics['psnr_std']))
    print("[EVAL] Mean SSIM = {:.6f} (+/-{:.4f})".format(metrics['ssim'], metrics['ssim_std']))
    print("[EVAL] Mean RMSE = {:.6f}".format(metrics['rmse']))
    print("[EVAL] Mean Rel. L2 = {:.6f}".format(metrics['relative_l2']))

    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("[SAVE] Metrics -> {}".format(metrics_path))

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
        "FNO Darcy Flow | PSNR={:.2f} dB | SSIM={:.4f} | Rel. L2={:.4f}".format(
            sample_metrics['psnr'], sample_metrics['ssim'], sample_metrics['relative_l2']),
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("[VIS] Saved visualization -> {}".format(vis_path))

    np.save(os.path.join(results_dir, "reconstruction.npy"), predictions[best_idx, 0])
    np.save(os.path.join(results_dir, "ground_truth.npy"), u_test[best_idx])
    np.save(os.path.join(results_dir, "input_coefficient.npy"), a_test[best_idx])

    return metrics


# ============================================================
# Main test logic
# ============================================================

def main():
    data_paths = [
        '/data/yjh/neuralop_fno_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl'
    ]

    # Separate outer vs inner data files
    outer_path = None
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("[ERROR] No outer data file found.")
        sys.exit(1)

    # Load outer data
    print("[LOAD] Loading outer data from: {}".format(outer_path))
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)

    print("[INFO] Outer function: {}".format(outer_data.get('func_name', 'unknown')))

    # Check for chained execution
    is_chained = len(inner_paths) > 0

    if is_chained:
        # Pattern 2: Chained execution
        print("[MODE] Chained execution detected.")
        agent_output = run_inversion(*outer_args, **outer_kwargs)

        inner_path = inner_paths[0]
        print("[LOAD] Loading inner data from: {}".format(inner_path))
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)

        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})