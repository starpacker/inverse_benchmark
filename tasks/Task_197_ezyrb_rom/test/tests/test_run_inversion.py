import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json

# Define working directories
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# INJECT REFEREE (evaluate_results and helper functions)
# ============================================================================

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

# ============================================================================
# MAIN TEST LOGIC
# ============================================================================

def main():
    data_paths = ['/data/yjh/ezyrb_rom_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    print("=" * 70)
    print("TEST: run_inversion")
    print("=" * 70)
    
    # Separate outer and inner data files
    outer_files = []
    inner_files = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename or 'parent_' in filename:
            inner_files.append(path)
        else:
            outer_files.append(path)
    
    print(f"[INFO] Outer files: {outer_files}")
    print(f"[INFO] Inner files: {inner_files}")
    
    if not outer_files:
        print("[ERROR] No outer data file found!")
        sys.exit(1)
    
    # Load outer (primary) data
    outer_path = outer_files[0]
    print(f"\n[INFO] Loading outer data from: {outer_path}")
    
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print(f"[INFO] Outer data keys: {outer_data.keys()}")
    
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"[INFO] Running agent's run_inversion...")
    
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"[ERROR] Agent run_inversion failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if we have inner data (chained execution)
    if inner_files:
        print(f"\n[INFO] Chained execution detected. Loading inner data...")
        inner_path = inner_files[0]
        
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        # agent_output should be callable
        if callable(agent_output):
            print(f"[INFO] Executing inner function...")
            try:
                final_result = agent_output(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"[ERROR] Inner function execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
        else:
            print("[ERROR] Expected callable from outer function, but got non-callable")
            sys.exit(1)
    else:
        # Direct execution - use outer output
        final_result = agent_output
        std_result = std_output
    
    # Now we need to evaluate the results
    # The run_inversion returns (predictions, rom_info)
    # We need test data (snapshots_test) to evaluate
    
    print("\n[INFO] Extracting predictions and rom_info...")
    
    # Handle tuple output
    if isinstance(final_result, tuple) and len(final_result) == 2:
        agent_predictions, agent_rom_info = final_result
    else:
        print(f"[ERROR] Unexpected output format: {type(final_result)}")
        sys.exit(1)
    
    if isinstance(std_result, tuple) and len(std_result) == 2:
        std_predictions, std_rom_info = std_result
    else:
        print(f"[ERROR] Unexpected standard output format: {type(std_result)}")
        sys.exit(1)
    
    # Extract parameters from args for evaluation
    # args = (params_train, snapshots_train, k_test_values, nx, ny)
    params_train = args[0]
    snapshots_train = args[1]
    k_test_values = args[2]
    nx = args[3]
    ny = args[4]
    
    print(f"[INFO] Grid size: {nx} x {ny}")
    print(f"[INFO] Number of test values: {len(k_test_values)}")
    print(f"[INFO] Agent predictions shape: {agent_predictions.shape}")
    print(f"[INFO] Standard predictions shape: {std_predictions.shape}")
    
    # For evaluation, we use the standard predictions as "ground truth"
    # since the standard code output represents the expected behavior
    # We compare agent predictions against standard predictions
    
    print("\n" + "=" * 70)
    print("EVALUATING AGENT OUTPUT")
    print("=" * 70)
    
    # Compute metrics comparing agent to standard
    agent_metrics = []
    for i, k_val in enumerate(k_test_values):
        gt = std_predictions[i]  # Use standard as ground truth
        pred = agent_predictions[i]
        
        gt_2d = gt.reshape(nx, ny)
        pred_2d = pred.reshape(nx, ny)
        
        psnr = compute_psnr(gt, pred)
        ssim = compute_ssim_2d(gt_2d, pred_2d)
        rmse = compute_rmse(gt, pred)
        rel_l2 = compute_relative_l2(gt, pred)
        
        agent_metrics.append({
            'k': float(k_val),
            'psnr': float(psnr),
            'ssim': float(ssim),
            'rmse': float(rmse),
            'relative_l2': float(rel_l2),
        })
    
    avg_psnr = np.mean([m['psnr'] for m in agent_metrics])
    avg_ssim = np.mean([m['ssim'] for m in agent_metrics])
    avg_rmse = np.mean([m['rmse'] for m in agent_metrics])
    avg_rel_l2 = np.mean([m['relative_l2'] for m in agent_metrics])
    
    print(f"\n[COMPARISON] Agent vs Standard:")
    print(f"  Average PSNR:        {avg_psnr:.2f} dB")
    print(f"  Average SSIM:        {avg_ssim:.4f}")
    print(f"  Average RMSE:        {avg_rmse:.6f}")
    print(f"  Average Relative L2: {avg_rel_l2:.6f}")
    
    # Also compute self-consistency metrics for the standard output
    # (just to show what the standard achieves)
    print(f"\n[INFO] Agent ROM info:")
    print(f"  n_modes:   {agent_rom_info.get('n_modes', 'N/A')}")
    print(f"  fit_time:  {agent_rom_info.get('fit_time', 'N/A'):.4f} sec")
    print(f"  n_train:   {agent_rom_info.get('n_train', 'N/A')}")
    
    print(f"\n[INFO] Standard ROM info:")
    print(f"  n_modes:   {std_rom_info.get('n_modes', 'N/A')}")
    print(f"  fit_time:  {std_rom_info.get('fit_time', 'N/A'):.4f} sec")
    print(f"  n_train:   {std_rom_info.get('n_train', 'N/A')}")
    
    # Determine success criteria
    # PSNR: Higher is better, SSIM: Higher is better (max 1.0)
    # For comparing agent to standard, we want high PSNR (> 40 dB is excellent)
    # and high SSIM (> 0.99 is excellent)
    
    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)
    
    # Check if agent output matches standard output closely
    # PSNR > 40 dB means very close match
    # SSIM > 0.99 means excellent structural similarity
    
    psnr_threshold = 40.0  # dB - very high similarity
    ssim_threshold = 0.99  # excellent structural match
    rel_l2_threshold = 0.01  # 1% relative error
    
    passed = True
    
    if avg_psnr < psnr_threshold:
        print(f"[WARNING] PSNR {avg_psnr:.2f} dB is below threshold {psnr_threshold} dB")
        if avg_psnr < 30.0:  # Significant difference
            passed = False
    else:
        print(f"[PASS] PSNR {avg_psnr:.2f} dB >= {psnr_threshold} dB")
    
    if avg_ssim < ssim_threshold:
        print(f"[WARNING] SSIM {avg_ssim:.4f} is below threshold {ssim_threshold}")
        if avg_ssim < 0.95:  # Significant difference
            passed = False
    else:
        print(f"[PASS] SSIM {avg_ssim:.4f} >= {ssim_threshold}")
    
    if avg_rel_l2 > rel_l2_threshold:
        print(f"[WARNING] Relative L2 {avg_rel_l2:.6f} exceeds threshold {rel_l2_threshold}")
        if avg_rel_l2 > 0.1:  # 10% error is too much
            passed = False
    else:
        print(f"[PASS] Relative L2 {avg_rel_l2:.6f} <= {rel_l2_threshold}")
    
    # Additional check: ensure shapes match
    if agent_predictions.shape != std_predictions.shape:
        print(f"[FAIL] Shape mismatch: Agent {agent_predictions.shape} vs Standard {std_predictions.shape}")
        passed = False
    else:
        print(f"[PASS] Output shapes match: {agent_predictions.shape}")
    
    # Check ROM info consistency
    if agent_rom_info.get('n_modes') != std_rom_info.get('n_modes'):
        print(f"[WARNING] n_modes differ: Agent {agent_rom_info.get('n_modes')} vs Standard {std_rom_info.get('n_modes')}")
    
    if agent_rom_info.get('n_train') != std_rom_info.get('n_train'):
        print(f"[WARNING] n_train differ: Agent {agent_rom_info.get('n_train')} vs Standard {std_rom_info.get('n_train')}")
    
    print("\n" + "=" * 70)
    
    if passed:
        print("[SUCCESS] Agent's run_inversion produces results consistent with standard.")
        print(f"Scores -> Agent PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, RelL2: {avg_rel_l2:.6f}")
        sys.exit(0)
    else:
        print("[FAILURE] Agent's run_inversion produces significantly different results.")
        print(f"Scores -> Agent PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, RelL2: {avg_rel_l2:.6f}")
        sys.exit(1)


if __name__ == "__main__":
    main()