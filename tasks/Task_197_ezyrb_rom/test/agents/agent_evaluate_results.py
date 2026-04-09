import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

import os

import json

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

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

def evaluate_results(snapshots_test, predictions, k_test_values, rom_info, nx, ny):
    """
    Evaluate reconstruction quality and generate visualizations.
    
    Parameters
    ----------
    snapshots_test : ndarray of shape (n_test, nx*ny)
        Ground truth temperature fields.
    predictions : ndarray of shape (n_test, nx*ny)
        ROM predictions.
    k_test_values : array-like
        Test parameter values.
    rom_info : dict
        Information about the ROM.
    nx, ny : int
        Grid resolution.
    
    Returns
    -------
    metrics_out : dict
        Comprehensive metrics dictionary.
    """
    all_metrics = []
    
    for i, k_val in enumerate(k_test_values):
        gt = snapshots_test[i]
        pred = predictions[i]
        
        gt_2d = gt.reshape(nx, ny)
        pred_2d = pred.reshape(nx, ny)
        
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
    
    avg_psnr = np.mean([m['psnr'] for m in all_metrics])
    avg_ssim = np.mean([m['ssim'] for m in all_metrics])
    avg_rmse = np.mean([m['rmse'] for m in all_metrics])
    avg_rel_l2 = np.mean([m['relative_l2'] for m in all_metrics])
    
    print(f"\n      Average: PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.4f}, "
          f"RMSE={avg_rmse:.6f}, relL2={avg_rel_l2:.6f}")
    
    sorted_by_psnr = sorted(range(len(all_metrics)),
                            key=lambda i: all_metrics[i]['psnr'])
    vis_idx = sorted_by_psnr[len(sorted_by_psnr) // 2]
    k_vis = k_test_values[vis_idx]
    
    gt_vis = snapshots_test[vis_idx]
    pred_vis = predictions[vis_idx]
    gt_2d = gt_vis.reshape(nx, ny)
    pred_2d = pred_vis.reshape(nx, ny)
    error_2d = np.abs(gt_2d - pred_2d)
    
    vis_metrics = {
        'psnr': all_metrics[vis_idx]['psnr'],
        'ssim': all_metrics[vis_idx]['ssim'],
        'rmse': all_metrics[vis_idx]['rmse'],
        'relative_l2': all_metrics[vis_idx]['relative_l2'],
        'n_train': rom_info['n_train'],
        'n_pod_modes': rom_info['n_modes'],
        'grid_size': f'{nx}x{ny}',
    }
    
    fig_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    _plot_results(gt_2d, pred_2d, error_2d, vis_metrics, k_vis, fig_path)
    
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt_2d)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), pred_2d)
    np.save(os.path.join(RESULTS_DIR, "all_gt.npy"), snapshots_test)
    np.save(os.path.join(RESULTS_DIR, "all_predictions.npy"), predictions)
    
    metrics_out = {
        "task": "ezyrb_rom",
        "method": "POD + RBF (EZyRB ReducedOrderModel)",
        "problem": "Parametric 2D heat conduction inverse problem",
        "description": "Reconstruct temperature field at unseen thermal conductivity from sparse snapshots",
        "grid_size": [nx, ny],
        "n_train_snapshots": rom_info['n_train'],
        "n_test_snapshots": len(k_test_values),
        "n_pod_modes": rom_info['n_modes'],
        "fit_time_sec": round(rom_info['fit_time'], 3),
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

def _plot_results(gt_2d, pred_2d, error_2d, metrics, k_test, save_path):
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

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(gt_2d.T, origin='lower', cmap='hot', aspect='equal',
                     vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
    ax1.set_title(f'(a) Ground Truth (k = {k_test:.2f})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    plt.colorbar(im1, ax=ax1, label='Temperature', shrink=0.8)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(pred_2d.T, origin='lower', cmap='hot', aspect='equal',
                     vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
    ax2.set_title(f'(b) ROM Prediction (POD + RBF)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    plt.colorbar(im2, ax=ax2, label='Temperature', shrink=0.8)

    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(error_2d.T, origin='lower', cmap='RdBu_r', aspect='equal',
                     extent=[0, 1, 0, 1])
    ax3.set_title('(c) Absolute Error', fontsize=14, fontweight='bold')
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('y', fontsize=12)
    plt.colorbar(im3, ax=ax3, label='|GT - Prediction|', shrink=0.8)

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
