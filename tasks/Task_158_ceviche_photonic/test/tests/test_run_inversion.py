import sys
import os
import dill
import numpy as np
import traceback
import json

# Import the agent's function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Inject the referee evaluation function
def evaluate_results(data, inversion_result, results_dir):
    """
    Compute metrics, save results, and create visualizations.
    
    Args:
        data: dict from load_and_preprocess_data
        inversion_result: dict from run_inversion
        results_dir: directory to save results
    
    Returns:
        dict of metrics
    """
    os.makedirs(results_dir, exist_ok=True)
    
    eps_gt = data['eps_gt']
    Ez_gt = data['Ez_gt']
    params = data['params']
    
    eps_opt = inversion_result['eps_opt']
    Ez_opt = inversion_result['Ez_opt']
    loss_history = inversion_result['loss_history']
    
    nx = params['nx']
    ny = params['ny']
    npml = params['npml']
    wg_width = params['wg_width']
    eps_min = params['eps_min']
    eps_max = params['eps_max']
    
    # Compute metrics
    print("\n[4] Computing metrics...")
    
    # Normalize permittivity to [0,1]
    eps_gt_n = (eps_gt - eps_min) / (eps_max - eps_min)
    eps_opt_n = (eps_opt - eps_min) / (eps_max - eps_min)
    
    # Structure metrics
    mse_s = float(np.mean((eps_gt_n - eps_opt_n)**2))
    psnr_s = float(10 * np.log10(1.0 / (mse_s + 1e-10)))
    ssim_s = float(ssim(eps_gt_n, eps_opt_n, data_range=1.0, win_size=3))
    cc_s = float(np.corrcoef(eps_gt_n.flatten(), eps_opt_n.flatten())[0, 1])
    
    # Field metrics
    Ez_gt_a = np.abs(Ez_gt)
    Ez_opt_a = np.abs(Ez_opt)
    Ez_gt_n2 = Ez_gt_a / (np.max(Ez_gt_a) + 1e-30)
    Ez_opt_n2 = Ez_opt_a / (np.max(Ez_opt_a) + 1e-30)
    
    cc_f = float(np.corrcoef(Ez_gt_n2.flatten(), Ez_opt_n2.flatten())[0, 1])
    mse_f = float(np.mean((Ez_gt_n2 - Ez_opt_n2)**2))
    psnr_f = float(10 * np.log10(1.0 / (mse_f + 1e-10)))
    ssim_f = float(ssim(Ez_gt_n2, Ez_opt_n2, data_range=1.0, win_size=3))
    
    # Transmission at a probe location (right side of waveguide)
    probe = np.zeros((nx, ny))
    cx = nx // 2
    hw = wg_width // 2
    probe_y = ny - npml - 2
    probe[cx - hw : cx + hw + 1, probe_y] = 1.0
    T_gt_val = float(np.sum(np.abs(Ez_gt * probe)**2))
    T_opt_val = float(np.sum(np.abs(Ez_opt * probe)**2))
    T_ratio = T_opt_val / (T_gt_val + 1e-30)
    
    metrics = {
        "PSNR": psnr_s,
        "SSIM": ssim_s,
        "structure_psnr_db": psnr_s,
        "structure_ssim": ssim_s,
        "structure_cc": cc_s,
        "structure_mse": mse_s,
        "field_psnr_db": psnr_f,
        "field_ssim": ssim_f,
        "field_cc": cc_f,
        "transmission_gt": T_gt_val,
        "transmission_opt": T_opt_val,
        "transmission_ratio": T_ratio,
    }
    
    for k, v in sorted(metrics.items()):
        print(f"    {k}: {v:.6f}")
    
    # Save outputs
    print("\n[5] Saving outputs...")
    np.save(os.path.join(results_dir, "gt_output.npy"), eps_gt)
    np.save(os.path.join(results_dir, "recon_output.npy"), eps_opt)
    np.save(os.path.join(results_dir, "gt_field.npy"), np.abs(Ez_gt))
    np.save(os.path.join(results_dir, "opt_field.npy"), np.abs(Ez_opt))
    np.save(os.path.join(results_dir, "loss_history.npy"), np.array(loss_history))
    
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"    Saved: {metrics_path}")
    
    # Create visualization
    print("\n[6] Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    eps_gt_d = (eps_gt - eps_min) / (eps_max - eps_min)
    eps_opt_d = (eps_opt - eps_min) / (eps_max - eps_min)
    
    # Panel 1: GT structure
    ax = axes[0, 0]
    im = ax.imshow(eps_gt_d.T, origin='lower', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_title('Ground Truth Dielectric Structure', fontsize=13, fontweight='bold')
    ax.set_xlabel('x (grid cells)')
    ax.set_ylabel('y (grid cells)')
    plt.colorbar(im, ax=ax, label='Normalized eps', shrink=0.8)
    
    # Panel 2: Optimized structure
    ax = axes[0, 1]
    im = ax.imshow(eps_opt_d.T, origin='lower', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_title(f'Optimized Structure (CC={metrics["structure_cc"]:.3f})',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('x (grid cells)')
    ax.set_ylabel('y (grid cells)')
    plt.colorbar(im, ax=ax, label='Normalized eps', shrink=0.8)
    
    # Panel 3: Field pattern (optimized)
    ax = axes[1, 0]
    Ez_opt_n = np.abs(Ez_opt) / (np.max(np.abs(Ez_opt)) + 1e-30)
    im = ax.imshow(Ez_opt_n.T, origin='lower', cmap='hot', vmin=0, vmax=1)
    ax.set_title(f'|Ez| Field (Optimized), Field CC={metrics["field_cc"]:.3f}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('x (grid cells)')
    ax.set_ylabel('y (grid cells)')
    plt.colorbar(im, ax=ax, label='|Ez| (normalized)', shrink=0.8)
    
    # Panel 4: Error map
    ax = axes[1, 1]
    error = np.abs(eps_gt_d - eps_opt_d)
    im = ax.imshow(error.T, origin='lower', cmap='viridis', vmin=0)
    ax.set_title(f'|eps_GT - eps_opt| Error (PSNR={metrics["structure_psnr_db"]:.1f} dB)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('x (grid cells)')
    ax.set_ylabel('y (grid cells)')
    plt.colorbar(im, ax=ax, label='|Delta eps|', shrink=0.8)
    
    wavelength = 1.55e-6
    plt.suptitle('Task 158: Photonic Inverse Design (FDFD, ceviche)\n'
                 f'Grid: {nx}x{ny}, lambda={wavelength*1e9:.0f} nm, '
                 f'SSIM={metrics["structure_ssim"]:.3f}',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    vis_path = os.path.join(results_dir, "vis_result.png")
    fig.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {vis_path}")
    
    # Convergence plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(loss_history, 'b-', linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Optimization Convergence', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    conv_path = os.path.join(results_dir, "convergence.png")
    fig.savefig(conv_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {conv_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Structure PSNR: {metrics['structure_psnr_db']:.2f} dB")
    print(f"  Structure SSIM: {metrics['structure_ssim']:.4f}")
    print(f"  Structure CC:   {metrics['structure_cc']:.4f}")
    print(f"  Field CC:       {metrics['field_cc']:.4f}")
    print(f"  Field SSIM:     {metrics['field_ssim']:.4f}")
    print(f"  Transmission ratio: {metrics['transmission_ratio']:.4f}")
    print("=" * 70)
    
    return metrics


def main():
    # Define data paths
    data_paths = ['/data/yjh/ceviche_photonic_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    if outer_data_path is None:
        print("ERROR: No outer data file found!")
        sys.exit(1)
    
    print(f"Loading outer data from: {outer_data_path}")
    
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract inputs
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Running agent's run_inversion with args={len(args)}, kwargs keys={list(kwargs.keys())}")
    
    # Run the agent's function
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"ERROR running agent's run_inversion: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if there are inner data files (chained execution pattern)
    if inner_data_paths:
        print(f"\nDetected chained execution pattern with {len(inner_data_paths)} inner file(s)")
        
        for inner_path in inner_data_paths:
            print(f"Loading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Execute the operator returned by run_inversion
            try:
                final_result = agent_output(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR running inner function: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Direct execution pattern
        final_result = agent_output
        std_result = std_output
    
    # Create results directories
    results_dir_agent = './results_agent'
    results_dir_std = './results_std'
    
    # Extract input data for evaluation
    # The first argument should be the 'data' dict
    input_data = args[0] if args else kwargs.get('data', None)
    
    if input_data is None:
        print("ERROR: Could not extract input data for evaluation")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("EVALUATING AGENT OUTPUT")
    print("=" * 70)
    
    try:
        metrics_agent = evaluate_results(input_data, final_result, results_dir_agent)
    except Exception as e:
        print(f"ERROR evaluating agent output: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("EVALUATING STANDARD OUTPUT")
    print("=" * 70)
    
    try:
        metrics_std = evaluate_results(input_data, std_result, results_dir_std)
    except Exception as e:
        print(f"ERROR evaluating standard output: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Compare metrics
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    # Primary metrics for comparison (higher is better)
    primary_metric = 'PSNR'
    score_agent = metrics_agent.get(primary_metric, 0)
    score_std = metrics_std.get(primary_metric, 0)
    
    print(f"Primary Metric ({primary_metric}):")
    print(f"  Agent:    {score_agent:.4f}")
    print(f"  Standard: {score_std:.4f}")
    
    # Also compare SSIM
    ssim_agent = metrics_agent.get('SSIM', 0)
    ssim_std = metrics_std.get('SSIM', 0)
    
    print(f"\nSSIM:")
    print(f"  Agent:    {ssim_agent:.4f}")
    print(f"  Standard: {ssim_std:.4f}")
    
    # Structure CC
    cc_agent = metrics_agent.get('structure_cc', 0)
    cc_std = metrics_std.get('structure_cc', 0)
    
    print(f"\nStructure CC:")
    print(f"  Agent:    {cc_agent:.4f}")
    print(f"  Standard: {cc_std:.4f}")
    
    # Determine success: allow 10% margin for PSNR (higher is better)
    # PSNR can vary more due to optimization randomness, so use a more lenient margin
    margin = 0.8  # 20% margin for optimization algorithms
    
    psnr_pass = score_agent >= score_std * margin
    ssim_pass = ssim_agent >= ssim_std * margin
    cc_pass = cc_agent >= cc_std * margin
    
    print(f"\n" + "=" * 70)
    print("PASS/FAIL STATUS")
    print("=" * 70)
    print(f"  PSNR test (>= {margin*100:.0f}% of std): {'PASS' if psnr_pass else 'FAIL'}")
    print(f"  SSIM test (>= {margin*100:.0f}% of std): {'PASS' if ssim_pass else 'FAIL'}")
    print(f"  CC test (>= {margin*100:.0f}% of std):   {'PASS' if cc_pass else 'FAIL'}")
    
    # Overall pass if at least 2 out of 3 metrics pass
    num_passed = sum([psnr_pass, ssim_pass, cc_pass])
    overall_pass = num_passed >= 2
    
    print(f"\nOverall: {'PASS' if overall_pass else 'FAIL'} ({num_passed}/3 metrics passed)")
    
    if overall_pass:
        print("\n✓ Agent's run_inversion performance is acceptable.")
        sys.exit(0)
    else:
        print("\n✗ Agent's run_inversion performance is degraded.")
        sys.exit(1)


if __name__ == '__main__':
    main()