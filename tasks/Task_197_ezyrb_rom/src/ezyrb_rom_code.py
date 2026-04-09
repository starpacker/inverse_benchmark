"""
ezyrb_rom - Reduced-Order Model Inverse Problem
================================================
Task: From sparse parameter snapshots, reconstruct full solution fields at new parameters
Method: POD (Proper Orthogonal Decomposition) + RBF interpolation via EZyRB
Repo: https://github.com/mathLab/EZyRB

Inverse problem: Given sparse parameter-snapshot pairs from a parametric 2D heat
conduction problem, predict the full temperature field at unseen parameter values
using a reduced-order model.

Usage:
    /data/yjh/ezyrb_rom_env/bin/python ezyrb_rom_code.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
import json
import time

from ezyrb import POD, RBF, GPR, Database, ReducedOrderModel

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Synthetic Data Generation: Parametric 2D Heat Conduction
# ---------------------------------------------------------------------------
def solve_heat_equation(k, nx=64, ny=64):
    """
    Solve steady-state 2D heat equation: -k * (d^2T/dx^2 + d^2T/dy^2) = f(x,y)
    on [0,1]x[0,1] with Dirichlet BCs using finite differences.

    Parameters
    ----------
    k : float
        Thermal conductivity parameter.
    nx, ny : int
        Grid resolution.

    Returns
    -------
    T : ndarray of shape (nx*ny,)
        Flattened temperature field.
    """
    dx = 1.0 / (nx + 1)
    dy = 1.0 / (ny + 1)

    # Interior grid points
    x = np.linspace(dx, 1.0 - dx, nx)
    y = np.linspace(dy, 1.0 - dy, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Source term: localized heat source
    f = 100.0 * np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / (2 * 0.05**2))
    # Add a k-dependent secondary source to make snapshots vary more with k
    f += 50.0 * k * np.sin(2 * np.pi * X) * np.sin(np.pi * Y)

    N = nx * ny
    # Build sparse-like coefficient matrix using direct indexing
    # For a 5-point stencil: -k*(T_{i-1,j} + T_{i+1,j} + T_{i,j-1} + T_{i,j+1} - 4*T_{i,j}) / h^2 = f
    # Assuming dx = dy = h
    h = dx
    coeff_diag = 4.0 * k / h**2
    coeff_off = -k / h**2

    # Use scipy sparse solver for efficiency
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve

    A = lil_matrix((N, N))
    rhs = f.flatten()

    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            A[idx, idx] = coeff_diag

            # Left neighbor
            if i > 0:
                A[idx, (i - 1) * ny + j] = coeff_off
            # Right neighbor
            if i < nx - 1:
                A[idx, (i + 1) * ny + j] = coeff_off
            # Bottom neighbor
            if j > 0:
                A[idx, i * ny + (j - 1)] = coeff_off
            # Top neighbor
            if j < ny - 1:
                A[idx, i * ny + (j + 1)] = coeff_off

    A = A.tocsr()
    T = spsolve(A, rhs)
    return T


def generate_snapshots(k_values, nx=64, ny=64):
    """
    Generate solution snapshots for a range of thermal conductivity values.

    Parameters
    ----------
    k_values : array-like
        Thermal conductivity parameter values.
    nx, ny : int
        Grid resolution.

    Returns
    -------
    params : ndarray of shape (n_samples, 1)
    snapshots : ndarray of shape (n_samples, nx*ny)
    """
    snapshots = []
    for k in k_values:
        T = solve_heat_equation(k, nx, ny)
        snapshots.append(T)

    params = np.array(k_values).reshape(-1, 1)
    snapshots = np.array(snapshots)
    return params, snapshots


# ---------------------------------------------------------------------------
# 2. Forward Operator
# ---------------------------------------------------------------------------
def forward_operator(k, nx=64, ny=64):
    """
    Forward operator: map thermal conductivity parameter to temperature field.
    This IS the parameter-to-solution map.
    """
    return solve_heat_equation(k, nx, ny)


# ---------------------------------------------------------------------------
# 3. Evaluation Metrics
# ---------------------------------------------------------------------------
def compute_relative_l2(gt, pred):
    """Relative L2 error: ||gt - pred||_2 / ||gt||_2"""
    return np.linalg.norm(gt - pred) / (np.linalg.norm(gt) + 1e-12)


def compute_psnr(gt, pred):
    """Peak Signal-to-Noise Ratio"""
    mse = np.mean((gt - pred) ** 2)
    if mse < 1e-20:
        return 100.0
    max_val = np.max(np.abs(gt))
    return 10.0 * np.log10(max_val**2 / mse)


def compute_ssim_2d(gt_2d, pred_2d):
    """Compute SSIM for 2D fields using skimage"""
    try:
        from skimage.metrics import structural_similarity
        data_range = gt_2d.max() - gt_2d.min()
        if data_range < 1e-12:
            data_range = 1.0
        return structural_similarity(gt_2d, pred_2d, data_range=data_range)
    except ImportError:
        # Fallback: simple SSIM approximation
        mu_gt = np.mean(gt_2d)
        mu_pred = np.mean(pred_2d)
        sig_gt = np.std(gt_2d)
        sig_pred = np.std(pred_2d)
        sig_cross = np.mean((gt_2d - mu_gt) * (pred_2d - mu_pred))
        C1 = (0.01 * (gt_2d.max() - gt_2d.min())) ** 2
        C2 = (0.03 * (gt_2d.max() - gt_2d.min())) ** 2
        ssim = ((2 * mu_gt * mu_pred + C1) * (2 * sig_cross + C2)) / \
               ((mu_gt**2 + mu_pred**2 + C1) * (sig_gt**2 + sig_pred**2 + C2))
        return float(ssim)


def compute_rmse(gt, pred):
    """Root Mean Square Error"""
    return np.sqrt(np.mean((gt - pred) ** 2))


# ---------------------------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------------------------
def plot_results(gt_2d, pred_2d, error_2d, metrics, k_test, save_path):
    """
    Create a 4-panel figure:
      (a) Ground truth field
      (b) ROM prediction
      (c) Absolute error map
      (d) Metrics summary
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    vmin = min(gt_2d.min(), pred_2d.min())
    vmax = max(gt_2d.max(), pred_2d.max())

    # (a) Ground Truth
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(gt_2d.T, origin='lower', cmap='hot', aspect='equal',
                     vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
    ax1.set_title(f'(a) Ground Truth (k = {k_test:.2f})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    plt.colorbar(im1, ax=ax1, label='Temperature', shrink=0.8)

    # (b) ROM Prediction
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(pred_2d.T, origin='lower', cmap='hot', aspect='equal',
                     vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
    ax2.set_title(f'(b) ROM Prediction (POD + RBF)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    plt.colorbar(im2, ax=ax2, label='Temperature', shrink=0.8)

    # (c) Error Map
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(error_2d.T, origin='lower', cmap='RdBu_r', aspect='equal',
                     extent=[0, 1, 0, 1])
    ax3.set_title('(c) Absolute Error', fontsize=14, fontweight='bold')
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('y', fontsize=12)
    plt.colorbar(im3, ax=ax3, label='|GT - Prediction|', shrink=0.8)

    # (d) Metrics Summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    metrics_text = (
        f"Reconstruction Metrics\n"
        f"{'='*35}\n\n"
        f"Test parameter:  k = {k_test:.3f}\n\n"
        f"PSNR:            {metrics['psnr']:.2f} dB\n"
        f"SSIM:            {metrics['ssim']:.4f}\n"
        f"RMSE:            {metrics['rmse']:.6f}\n"
        f"Relative L2:     {metrics['relative_l2']:.6f}\n\n"
        f"Training snapshots:  {metrics['n_train']}\n"
        f"POD modes used:      {metrics['n_pod_modes']}\n"
        f"Grid resolution:     {metrics['grid_size']}\n"
        f"Interpolation:       RBF\n"
    )
    ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes,
             fontsize=13, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
                       edgecolor='gray', alpha=0.9))
    ax4.set_title('(d) Evaluation Metrics', fontsize=14, fontweight='bold')

    plt.suptitle('EZyRB: Reduced-Order Model for 2D Heat Conduction\n'
                 'Inverse Problem: Reconstruct temperature field from sparse parameter snapshots',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Figure saved to {save_path}")


# ---------------------------------------------------------------------------
# 5. Main Pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("EZyRB ROM — Reduced-Order Model Inverse Problem")
    print("Parametric 2D Heat Conduction: k → Temperature field")
    print("=" * 70)

    NX, NY = 64, 64
    np.random.seed(42)

    # ---- Generate training + test data ----
    # Training: 15 parameter values
    k_train = np.linspace(0.5, 5.0, 15)
    # Test: 5 unseen parameter values (interleaved)
    k_test = np.array([0.8, 1.75, 2.6, 3.35, 4.25])

    print(f"\n[1/5] Generating training snapshots (k = {k_train[0]:.1f} to {k_train[-1]:.1f}, "
          f"n={len(k_train)})...")
    t0 = time.time()
    params_train, snapshots_train = generate_snapshots(k_train, NX, NY)
    print(f"      Training data: {snapshots_train.shape} in {time.time()-t0:.1f}s")

    print(f"\n[2/5] Generating test snapshots (n={len(k_test)})...")
    t0 = time.time()
    params_test, snapshots_test = generate_snapshots(k_test, NX, NY)
    print(f"      Test data: {snapshots_test.shape} in {time.time()-t0:.1f}s")

    # ---- Build ROM using EZyRB ----
    print("\n[3/5] Building Reduced-Order Model (POD + RBF)...")
    db = Database(params_train, snapshots_train)
    pod = POD('svd')
    rbf = RBF()
    rom = ReducedOrderModel(db, pod, rbf)
    t0 = time.time()
    rom.fit()
    fit_time = time.time() - t0
    print(f"      ROM fitted in {fit_time:.2f}s")

    # Get number of POD modes
    n_modes = rom.reduction.singular_values.shape[0]
    print(f"      POD modes: {n_modes}")

    # ---- Evaluate on test parameters ----
    print("\n[4/5] Evaluating ROM predictions on test parameters...")
    all_metrics = []
    best_idx = 0
    best_psnr = -1

    for i, k_val in enumerate(k_test):
        gt = snapshots_test[i]
        pred = rom.predict([k_val]).flatten()

        gt_2d = gt.reshape(NX, NY)
        pred_2d = pred.reshape(NX, NY)

        psnr = compute_psnr(gt, pred)
        ssim = compute_ssim_2d(gt_2d, pred_2d)
        rmse = compute_rmse(gt, pred)
        rel_l2 = compute_relative_l2(gt, pred)

        m = {
            'k': float(k_val),
            'psnr': float(psnr),
            'ssim': float(ssim),
            'rmse': float(rmse),
            'relative_l2': float(rel_l2),
        }
        all_metrics.append(m)
        print(f"      k={k_val:.2f}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}, "
              f"RMSE={rmse:.6f}, relL2={rel_l2:.6f}")

        if psnr > best_psnr:
            best_psnr = psnr
            best_idx = i

    # Aggregate metrics (average over test set)
    avg_psnr = np.mean([m['psnr'] for m in all_metrics])
    avg_ssim = np.mean([m['ssim'] for m in all_metrics])
    avg_rmse = np.mean([m['rmse'] for m in all_metrics])
    avg_rel_l2 = np.mean([m['relative_l2'] for m in all_metrics])

    print(f"\n      Average: PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.4f}, "
          f"RMSE={avg_rmse:.6f}, relL2={avg_rel_l2:.6f}")

    # ---- Visualization (use median test case for figure) ----
    print("\n[5/5] Generating visualization...")
    # Pick the median PSNR test case for representative visualization
    sorted_by_psnr = sorted(range(len(all_metrics)),
                            key=lambda i: all_metrics[i]['psnr'])
    vis_idx = sorted_by_psnr[len(sorted_by_psnr) // 2]
    k_vis = k_test[vis_idx]

    gt_vis = snapshots_test[vis_idx]
    pred_vis = rom.predict([k_vis]).flatten()
    gt_2d = gt_vis.reshape(NX, NY)
    pred_2d = pred_vis.reshape(NX, NY)
    error_2d = np.abs(gt_2d - pred_2d)

    vis_metrics = {
        'psnr': all_metrics[vis_idx]['psnr'],
        'ssim': all_metrics[vis_idx]['ssim'],
        'rmse': all_metrics[vis_idx]['rmse'],
        'relative_l2': all_metrics[vis_idx]['relative_l2'],
        'n_train': len(k_train),
        'n_pod_modes': int(n_modes),
        'grid_size': f'{NX}x{NY}',
    }

    fig_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    plot_results(gt_2d, pred_2d, error_2d, vis_metrics, k_vis, fig_path)

    # ---- Save outputs ----
    # Ground truth and reconstruction for the visualized test case
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt_2d)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), pred_2d)

    # Save all test ground truths and predictions
    np.save(os.path.join(RESULTS_DIR, "all_gt.npy"), snapshots_test)
    preds_all = np.array([rom.predict([k]).flatten() for k in k_test])
    np.save(os.path.join(RESULTS_DIR, "all_predictions.npy"), preds_all)

    # Metrics JSON
    metrics_out = {
        "task": "ezyrb_rom",
        "method": "POD + RBF (EZyRB ReducedOrderModel)",
        "problem": "Parametric 2D heat conduction inverse problem",
        "description": "Reconstruct temperature field at unseen thermal conductivity from sparse snapshots",
        "grid_size": [NX, NY],
        "n_train_snapshots": len(k_train),
        "n_test_snapshots": len(k_test),
        "n_pod_modes": int(n_modes),
        "fit_time_sec": round(fit_time, 3),
        "psnr": round(float(avg_psnr), 4),
        "ssim": round(float(avg_ssim), 4),
        "rmse": round(float(avg_rmse), 6),
        "relative_l2": round(float(avg_rel_l2), 6),
        "per_test_metrics": all_metrics,
    }
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n[INFO] Metrics saved to {metrics_path}")

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  PSNR:        {avg_psnr:.2f} dB")
    print(f"  SSIM:        {avg_ssim:.4f}")
    print(f"  RMSE:        {avg_rmse:.6f}")
    print(f"  Relative L2: {avg_rel_l2:.6f}")
    print(f"  Figure:      {fig_path}")
    print(f"  Metrics:     {metrics_path}")
    print("=" * 70)

    return metrics_out


if __name__ == "__main__":
    main()
