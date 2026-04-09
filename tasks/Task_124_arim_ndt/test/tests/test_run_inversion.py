import sys
import os
import dill
import numpy as np
import traceback
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, maximum_filter, label

# Import the agent's function
from agent_run_inversion import run_inversion


def evaluate_results(data_dict, result_dict):
    """
    Evaluate reconstruction results and save outputs.
    
    Computes PSNR, SSIM, defect position error, and generates visualizations.
    
    Parameters
    ----------
    data_dict : dict
        Data dictionary from load_and_preprocess_data.
    result_dict : dict
        Result dictionary from run_inversion.
    
    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics.
    """
    from skimage.metrics import structural_similarity
    
    gt_map = data_dict['gt_map']
    recon_norm = result_dict['recon_image']
    defects_m = data_dict['defects_m']
    defects_mm = data_dict['defects_mm']
    x_grid = data_dict['x_grid']
    z_grid = data_dict['z_grid']
    results_dir = data_dict['results_dir']
    recon_time = result_dict['recon_time']
    
    print("\n[3/4] Computing metrics...")
    
    # PSNR
    mse = np.mean((gt_map - recon_norm) ** 2)
    if mse < 1e-20:
        psnr = 100.0
    else:
        data_range = gt_map.max() - gt_map.min()
        psnr = 10 * np.log10(data_range ** 2 / mse)
    
    # SSIM
    ssim = structural_similarity(gt_map, recon_norm, data_range=gt_map.max() - gt_map.min())
    
    # Defect position error
    n_defects = len(defects_m)
    
    # Find local maxima
    filtered = maximum_filter(recon_norm, size=7)
    local_max = (recon_norm == filtered) & (recon_norm > 0.3 * recon_norm.max())
    labeled, n_found = label(local_max)
    
    # Centroid of each region
    peaks = []
    for i in range(1, n_found + 1):
        mask = labeled == i
        coords = np.argwhere(mask)
        iz_mean = coords[:, 0].mean()
        ix_mean = coords[:, 1].mean()
        amplitude = recon_norm[mask].max()
        peaks.append((iz_mean, ix_mean, amplitude))
    
    # Sort by amplitude descending, take top-N
    peaks.sort(key=lambda p: -p[2])
    peaks = peaks[:n_defects]
    
    # Convert to physical coordinates
    peak_positions = []
    for iz, ix, _ in peaks:
        iz_int = int(round(iz))
        ix_int = int(round(ix))
        iz_int = np.clip(iz_int, 0, len(z_grid) - 1)
        ix_int = np.clip(ix_int, 0, len(x_grid) - 1)
        peak_positions.append((x_grid[ix_int], z_grid[iz_int]))
    
    # Greedy nearest-neighbor matching
    gt_remaining = list(defects_m)
    total_error = 0.0
    matched = 0
    for px, pz in peak_positions:
        if not gt_remaining:
            break
        dists = [np.sqrt((px - gx) ** 2 + (pz - gz) ** 2) for gx, gz in gt_remaining]
        best_idx = np.argmin(dists)
        total_error += dists[best_idx]
        gt_remaining.pop(best_idx)
        matched += 1
    
    if matched == 0:
        pos_error_mm = float('inf')
    else:
        pos_error_mm = (total_error / matched) * 1000
    
    print(f"  PSNR:  {psnr:.2f} dB")
    print(f"  SSIM:  {ssim:.4f}")
    print(f"  Mean position error: {pos_error_mm:.2f} mm "
          f"({matched}/{len(defects_m)} defects matched)")
    
    metrics = {
        "task": "arim_ndt",
        "method": "Total Focusing Method (TFM)",
        "psnr_db": round(psnr, 2),
        "ssim": round(ssim, 4),
        "mean_position_error_mm": round(pos_error_mm, 2),
        "defects_matched": matched,
        "defects_total": len(defects_m),
        "n_elements": data_dict['n_elements'],
        "pitch_mm": data_dict['pitch'] * 1e3,
        "frequency_mhz": data_dict['freq'] / 1e6,
        "sound_speed_m_s": data_dict['c_sound'],
        "grid_nx": data_dict['nx'],
        "grid_nz": data_dict['nz'],
        "snr_db": data_dict['snr_db'],
        "reconstruction_time_s": round(recon_time, 1),
    }
    
    # Save outputs
    print("\n[4/4] Saving results...")
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, "ground_truth.npy"), gt_map)
    np.save(os.path.join(results_dir, "reconstruction.npy"), recon_norm)
    print(f"  Saved ground_truth.npy  shape={gt_map.shape}")
    print(f"  Saved reconstruction.npy  shape={recon_norm.shape}")
    
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics.json")
    
    # Visualization
    x_mm = x_grid * 1e3
    z_mm = z_grid * 1e3
    extent = [x_mm.min(), x_mm.max(), z_mm.max(), z_mm.min()]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    
    # Ground Truth
    ax = axes[0]
    im0 = ax.imshow(gt_map, extent=extent, cmap='hot', aspect='auto',
                    vmin=0, vmax=1)
    for dx_mm, dz_mm in defects_mm:
        ax.plot(dx_mm, dz_mm, 'c+', markersize=12, markeredgewidth=2)
    ax.set_title("Ground Truth\n(Gaussian defect map)", fontsize=13)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    plt.colorbar(im0, ax=ax, shrink=0.8)
    
    # TFM Reconstruction
    ax = axes[1]
    im1 = ax.imshow(recon_norm, extent=extent, cmap='hot', aspect='auto',
                    vmin=0, vmax=1)
    ax.set_title("TFM Reconstruction", fontsize=13)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    plt.colorbar(im1, ax=ax, shrink=0.8)
    
    # Overlay
    ax = axes[2]
    im2 = ax.imshow(recon_norm, extent=extent, cmap='hot', aspect='auto',
                    vmin=0, vmax=1)
    for dx_mm, dz_mm in defects_mm:
        ax.plot(dx_mm, dz_mm, 'c+', markersize=14, markeredgewidth=2,
                label='True defect')
    ax.set_title("Overlay (defects marked)", fontsize=13)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    plt.colorbar(im2, ax=ax, shrink=0.8)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=10)
    
    # Metrics text
    metrics_text = (f"PSNR: {metrics['psnr_db']:.2f} dB | "
                    f"SSIM: {metrics['ssim']:.4f} | "
                    f"Pos. Error: {metrics['mean_position_error_mm']:.2f} mm")
    fig.suptitle(f"NDT Ultrasonic TFM Imaging\n{metrics_text}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(results_dir, "reconstruction_result.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved visualization to {save_path}")
    
    print("\n" + "=" * 60)
    print("DONE. All results saved to:", results_dir)
    print("=" * 60)
    print(json.dumps(metrics, indent=2))
    
    return metrics


def main():
    data_paths = ['/data/yjh/arim_ndt_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p
    
    if outer_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)
    
    # Load outer data
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    func_name = outer_data.get('func_name', 'run_inversion')
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Function: {func_name}")
    print(f"Args count: {len(args)}, Kwargs keys: {list(kwargs.keys())}")
    
    # Determine execution pattern
    if len(inner_paths) > 0:
        # Pattern 2: Chained Execution
        print("\n=== Pattern 2: Chained Execution ===")
        print(f"Found {len(inner_paths)} inner data file(s)")
        
        # Step 1: Run outer function to get operator
        try:
            agent_operator = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"ERROR running outer function: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Step 2: Load inner data and run
        inner_path = inner_paths[0]
        print(f"Loading inner data from: {inner_path}")
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        try:
            agent_result = agent_operator(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR running inner function: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Pattern 1: Direct Execution
        print("\n=== Pattern 1: Direct Execution ===")
        
        try:
            agent_result = run_inversion(*args, **kwargs)
        except Exception as e:
            print(f"ERROR running run_inversion: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        std_result = std_output
    
    # Now evaluate both agent and standard results
    # The evaluate_results function needs (data_dict, result_dict)
    # data_dict is the input (args[0]), result_dict is the output
    
    if len(args) > 0:
        data_dict = args[0]
    elif 'data_dict' in kwargs:
        data_dict = kwargs['data_dict']
    else:
        print("ERROR: Cannot find data_dict input")
        sys.exit(1)
    
    # Ensure results_dir exists
    if 'results_dir' not in data_dict or not data_dict['results_dir']:
        data_dict['results_dir'] = '/tmp/ndt_results_test'
    os.makedirs(data_dict['results_dir'], exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Evaluating AGENT result...")
    print("=" * 60)
    
    try:
        # Use a separate results_dir for agent
        original_results_dir = data_dict['results_dir']
        agent_results_dir = os.path.join(original_results_dir, 'agent')
        os.makedirs(agent_results_dir, exist_ok=True)
        data_dict['results_dir'] = agent_results_dir
        
        metrics_agent = evaluate_results(data_dict, agent_result)
    except Exception as e:
        print(f"ERROR evaluating agent result: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Evaluating STANDARD result...")
    print("=" * 60)
    
    try:
        std_results_dir = os.path.join(original_results_dir, 'standard')
        os.makedirs(std_results_dir, exist_ok=True)
        data_dict['results_dir'] = std_results_dir
        
        metrics_std = evaluate_results(data_dict, std_result)
    except Exception as e:
        print(f"ERROR evaluating standard result: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Restore
    data_dict['results_dir'] = original_results_dir
    
    # Extract key metrics for comparison
    psnr_agent = metrics_agent.get('psnr_db', 0.0)
    psnr_std = metrics_std.get('psnr_db', 0.0)
    ssim_agent = metrics_agent.get('ssim', 0.0)
    ssim_std = metrics_std.get('ssim', 0.0)
    pos_err_agent = metrics_agent.get('mean_position_error_mm', float('inf'))
    pos_err_std = metrics_std.get('mean_position_error_mm', float('inf'))
    defects_matched_agent = metrics_agent.get('defects_matched', 0)
    defects_matched_std = metrics_std.get('defects_matched', 0)
    
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  PSNR  -> Agent: {psnr_agent:.2f} dB, Standard: {psnr_std:.2f} dB")
    print(f"  SSIM  -> Agent: {ssim_agent:.4f}, Standard: {ssim_std:.4f}")
    print(f"  Pos Error -> Agent: {pos_err_agent:.2f} mm, Standard: {pos_err_std:.2f} mm")
    print(f"  Defects Matched -> Agent: {defects_matched_agent}, Standard: {defects_matched_std}")
    
    # Determine pass/fail
    # PSNR and SSIM: higher is better (allow 10% degradation margin)
    # Position error: lower is better (allow 10% degradation margin)
    
    passed = True
    reasons = []
    
    # PSNR check: agent should be at least 90% of standard
    if psnr_std > 0:
        psnr_ratio = psnr_agent / psnr_std
        print(f"\n  PSNR ratio (agent/std): {psnr_ratio:.4f}")
        if psnr_ratio < 0.90:
            passed = False
            reasons.append(f"PSNR degraded: {psnr_agent:.2f} vs {psnr_std:.2f} (ratio={psnr_ratio:.4f})")
    
    # SSIM check: agent should be at least 90% of standard
    if ssim_std > 0:
        ssim_ratio = ssim_agent / ssim_std
        print(f"  SSIM ratio (agent/std): {ssim_ratio:.4f}")
        if ssim_ratio < 0.90:
            passed = False
            reasons.append(f"SSIM degraded: {ssim_agent:.4f} vs {ssim_std:.4f} (ratio={ssim_ratio:.4f})")
    
    # Position error check: agent should not be more than 110% of standard (lower is better)
    if pos_err_std > 0 and pos_err_std != float('inf'):
        pos_ratio = pos_err_agent / pos_err_std
        print(f"  Position error ratio (agent/std): {pos_ratio:.4f}")
        if pos_ratio > 1.10:
            # Only fail if the absolute difference is also significant
            if pos_err_agent - pos_err_std > 1.0:  # more than 1mm difference
                passed = False
                reasons.append(f"Position error increased: {pos_err_agent:.2f} vs {pos_err_std:.2f} mm (ratio={pos_ratio:.4f})")
    
    # Defects matched check
    if defects_matched_agent < defects_matched_std:
        passed = False
        reasons.append(f"Fewer defects matched: {defects_matched_agent} vs {defects_matched_std}")
    
    print(f"\nScores -> Agent PSNR: {psnr_agent:.2f}, Standard PSNR: {psnr_std:.2f}")
    print(f"Scores -> Agent SSIM: {ssim_agent:.4f}, Standard SSIM: {ssim_std:.4f}")
    
    if passed:
        print("\n*** TEST PASSED: Agent performance is acceptable. ***")
        sys.exit(0)
    else:
        print("\n*** TEST FAILED: Agent performance degraded significantly. ***")
        for r in reasons:
            print(f"  - {r}")
        sys.exit(1)


if __name__ == '__main__':
    main()