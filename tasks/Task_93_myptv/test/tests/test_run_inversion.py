import sys
import os
import dill
import numpy as np
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

# Import the target function from agent
from agent_run_inversion import run_inversion

# ============================================================================
# INJECTED REFEREE CODE (from Reference B)
# ============================================================================
def evaluate_results(gt_3d, recon_3d, n_views, reproj_errors, cameras, config,
                     results_dir=None):
    """
    Evaluate reconstruction quality and optionally save/visualize results.
    
    Args:
        gt_3d: (N, 3) ground truth positions
        recon_3d: (N, 3) reconstructed positions (may contain NaN)
        n_views: (N,) number of cameras that saw each particle
        reproj_errors: (N,) reprojection errors
        cameras: List of camera dictionaries
        config: Configuration dictionary
        results_dir: Directory to save results (None to skip saving)
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    vol_min = config['vol_min']
    vol_max = config['vol_max']
    n_cameras = config['n_cameras']
    
    # Mask for successfully reconstructed particles (≥2 views)
    valid = ~np.isnan(recon_3d[:, 0])
    n_valid = np.sum(valid)
    n_total = len(gt_3d)

    if n_valid == 0:
        metrics = {
            "rmse_3d_mm": float('inf'),
            "mean_error_mm": float('inf'),
            "median_error_mm": float('inf'),
            "max_error_mm": float('inf'),
            "correlation_x": 0.0,
            "correlation_y": 0.0,
            "correlation_z": 0.0,
            "correlation_mean": 0.0,
            "success_rate": 0.0,
            "n_reconstructed": 0,
            "n_total": n_total,
            "psnr_db": 0.0,
        }
    else:
        gt_valid = gt_3d[valid]
        recon_valid = recon_3d[valid]

        # 3D position errors
        errors_3d = np.linalg.norm(gt_valid - recon_valid, axis=1)
        rmse = np.sqrt(np.mean(errors_3d ** 2))
        mean_err = np.mean(errors_3d)
        median_err = np.median(errors_3d)
        max_err = np.max(errors_3d)

        # Per-axis correlation coefficients
        cc_x = np.corrcoef(gt_valid[:, 0], recon_valid[:, 0])[0, 1]
        cc_y = np.corrcoef(gt_valid[:, 1], recon_valid[:, 1])[0, 1]
        cc_z = np.corrcoef(gt_valid[:, 2], recon_valid[:, 2])[0, 1]
        cc_mean = (cc_x + cc_y + cc_z) / 3.0

        # Success rate
        success_rate = n_valid / n_total

        # PSNR: treat volume diagonal as signal range
        vol_diag = np.linalg.norm(vol_max - vol_min)
        mse = np.mean(errors_3d ** 2)
        psnr = 10.0 * np.log10(vol_diag ** 2 / mse) if mse > 0 else float('inf')

        metrics = {
            "rmse_3d_mm": float(np.round(rmse, 4)),
            "mean_error_mm": float(np.round(mean_err, 4)),
            "median_error_mm": float(np.round(median_err, 4)),
            "max_error_mm": float(np.round(max_err, 4)),
            "correlation_x": float(np.round(cc_x, 6)),
            "correlation_y": float(np.round(cc_y, 6)),
            "correlation_z": float(np.round(cc_z, 6)),
            "correlation_mean": float(np.round(cc_mean, 6)),
            "success_rate": float(np.round(success_rate, 4)),
            "n_reconstructed": int(n_valid),
            "n_total": int(n_total),
            "psnr_db": float(np.round(psnr, 2)),
        }

    # Optionally save results and visualize
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics
        metrics_path = os.path.join(results_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[SAVE] Metrics → {metrics_path}")

        # Save ground truth and reconstruction as .npy
        np.save(os.path.join(results_dir, "ground_truth.npy"), gt_3d)
        np.save(os.path.join(results_dir, "reconstruction.npy"), recon_3d)
        print(f"[SAVE] GT shape: {gt_3d.shape}")
        print(f"[SAVE] Recon shape: {recon_3d.shape}")

        # Visualize
        vis_path = os.path.join(results_dir, "reconstruction_result.png")
        _visualize_results(gt_3d, recon_3d, n_views, reproj_errors,
                           cameras, metrics, config, vis_path)

    return metrics

def _visualize_results(gt_3d, recon_3d, n_views, reproj_errors,
                       cameras, metrics, config, save_path):
    """
    Create comprehensive visualization.
    """
    vol_min = config['vol_min']
    vol_max = config['vol_max']
    n_cameras = config['n_cameras']
    
    valid = ~np.isnan(recon_3d[:, 0])
    gt_v = gt_3d[valid]
    rc_v = recon_3d[valid]
    errors_3d = np.linalg.norm(gt_v - rc_v, axis=1)

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(
        "3D Particle Tracking Velocimetry — Multi-Camera Triangulation\n"
        f"RMSE={metrics['rmse_3d_mm']:.3f} mm  |  "
        f"CC={metrics['correlation_mean']:.4f}  |  "
        f"PSNR={metrics['psnr_db']:.1f} dB  |  "
        f"Success={metrics['success_rate']*100:.1f}%",
        fontsize=14, fontweight='bold'
    )

    # (a) 3D scatter: GT (blue) vs Recon (red)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(gt_v[:, 0], gt_v[:, 1], gt_v[:, 2],
                c='blue', s=8, alpha=0.5, label='Ground Truth')
    ax1.scatter(rc_v[:, 0], rc_v[:, 1], rc_v[:, 2],
                c='red', s=8, alpha=0.5, label='Reconstructed')
    for cam in cameras:
        cp = cam['cam_pos']
        ax1.scatter(*cp, c='green', s=100, marker='^', zorder=5)
    ax1.set_xlabel('X [mm]')
    ax1.set_ylabel('Y [mm]')
    ax1.set_zlabel('Z [mm]')
    ax1.set_title('3D Positions: GT vs Recon')
    ax1.legend(fontsize=8)

    # (b) X-axis correlation
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(gt_v[:, 0], rc_v[:, 0], s=10, alpha=0.6, c='steelblue')
    lim = [vol_min[0] - 5, vol_max[0] + 5]
    ax2.plot(lim, lim, 'k--', lw=1)
    ax2.set_xlabel('GT X [mm]')
    ax2.set_ylabel('Recon X [mm]')
    ax2.set_title(f'X Correlation (CC={metrics["correlation_x"]:.4f})')
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # (c) Y-axis correlation
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(gt_v[:, 1], rc_v[:, 1], s=10, alpha=0.6, c='coral')
    lim_y = [vol_min[1] - 5, vol_max[1] + 5]
    ax3.plot(lim_y, lim_y, 'k--', lw=1)
    ax3.set_xlabel('GT Y [mm]')
    ax3.set_ylabel('Recon Y [mm]')
    ax3.set_title(f'Y Correlation (CC={metrics["correlation_y"]:.4f})')
    ax3.set_xlim(lim_y)
    ax3.set_ylim(lim_y)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # (d) Z-axis correlation
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(gt_v[:, 2], rc_v[:, 2], s=10, alpha=0.6, c='green')
    lim_z = [vol_min[2] - 5, vol_max[2] + 5]
    ax4.plot(lim_z, lim_z, 'k--', lw=1)
    ax4.set_xlabel('GT Z [mm]')
    ax4.set_ylabel('Recon Z [mm]')
    ax4.set_title(f'Z Correlation (CC={metrics["correlation_z"]:.4f})')
    ax4.set_xlim(lim_z)
    ax4.set_ylim(lim_z)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)

    # (e) 3D error histogram
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(errors_3d, bins=40, color='steelblue', edgecolor='black',
             alpha=0.7)
    ax5.axvline(metrics['rmse_3d_mm'], color='red', ls='--',
                label=f'RMSE={metrics["rmse_3d_mm"]:.3f} mm')
    ax5.axvline(metrics['median_error_mm'], color='orange', ls='--',
                label=f'Median={metrics["median_error_mm"]:.3f} mm')
    ax5.set_xlabel('3D Position Error [mm]')
    ax5.set_ylabel('Count')
    ax5.set_title('Error Distribution')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # (f) Number of views histogram
    ax6 = fig.add_subplot(2, 3, 6)
    view_counts = n_views[valid]
    bins_v = np.arange(0.5, n_cameras + 1.5, 1)
    ax6.hist(view_counts, bins=bins_v, color='mediumpurple',
             edgecolor='black', alpha=0.7)
    ax6.set_xlabel('Number of Camera Views')
    ax6.set_ylabel('Count')
    ax6.set_title('Views per Particle')
    ax6.set_xticks(range(1, n_cameras + 1))
    ax6.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")

# ============================================================================
# END OF INJECTED REFEREE CODE
# ============================================================================


def main():
    data_paths = ['/data/yjh/myptv_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    print(f"[INFO] Outer data path: {outer_data_path}")
    print(f"[INFO] Inner data paths: {inner_data_paths}")
    
    # Load outer data
    if outer_data_path is None or not os.path.exists(outer_data_path):
        print("[ERROR] Outer data file not found!")
        sys.exit(1)
    
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data: {outer_data.keys()}")
    except Exception as e:
        print(f"[ERROR] Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"[INFO] Args count: {len(args)}")
    print(f"[INFO] Kwargs keys: {list(kwargs.keys())}")
    
    # Run the agent function
    try:
        agent_output = run_inversion(*args, **kwargs)
        print("[INFO] Agent function executed successfully")
    except Exception as e:
        print(f"[ERROR] Agent function failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if chained execution is needed
    if len(inner_data_paths) > 0 and callable(agent_output):
        # Chained execution pattern
        print("[INFO] Detected chained execution pattern")
        for inner_path in inner_data_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                
                final_agent_result = agent_output(*inner_args, **inner_kwargs)
                final_std_result = inner_data.get('output', None)
                print(f"[INFO] Executed inner function from: {inner_path}")
            except Exception as e:
                print(f"[ERROR] Failed inner execution: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Direct execution pattern
        print("[INFO] Direct execution pattern")
        final_agent_result = agent_output
        final_std_result = std_output
    
    # Parse results (both should be tuples: recon_3d, n_views, reproj_errors)
    if isinstance(final_agent_result, tuple) and len(final_agent_result) >= 3:
        agent_recon_3d, agent_n_views, agent_reproj_errors = final_agent_result[:3]
    else:
        print("[ERROR] Agent output format unexpected")
        sys.exit(1)
    
    if isinstance(final_std_result, tuple) and len(final_std_result) >= 3:
        std_recon_3d, std_n_views, std_reproj_errors = final_std_result[:3]
    else:
        print("[ERROR] Standard output format unexpected")
        sys.exit(1)
    
    # We need gt_3d, cameras, and config for evaluation
    # These should be in the input args
    # args structure: (detections, visibility, cameras)
    detections = args[0] if len(args) > 0 else kwargs.get('detections')
    visibility = args[1] if len(args) > 1 else kwargs.get('visibility')
    cameras = args[2] if len(args) > 2 else kwargs.get('cameras')
    
    # For evaluation, we need gt_3d and config
    # Since this is an inversion problem, we don't have direct gt_3d in the inputs
    # We'll compare the agent reconstruction against the standard reconstruction
    # and use the standard as the "ground truth" reference
    
    # Create a minimal config for evaluation
    # Extract volume bounds from cameras or use defaults
    all_positions = []
    for cam in cameras:
        if 'cam_pos' in cam:
            all_positions.append(cam['cam_pos'])
    
    if all_positions:
        all_pos = np.array(all_positions)
        vol_min = np.array([-50, -50, 0])  # Default reasonable values
        vol_max = np.array([50, 50, 100])
    else:
        vol_min = np.array([-50, -50, 0])
        vol_max = np.array([50, 50, 100])
    
    config = {
        'vol_min': vol_min,
        'vol_max': vol_max,
        'n_cameras': len(cameras)
    }
    
    # Use standard reconstruction as ground truth for comparison
    # Evaluate agent against standard
    print("\n[INFO] Evaluating agent reconstruction against standard reconstruction...")
    
    # For metrics, we treat std_recon_3d as the reference (ground truth)
    # and compare agent_recon_3d to it
    
    # Extract valid points from both
    valid_std = ~np.isnan(std_recon_3d[:, 0])
    valid_agent = ~np.isnan(agent_recon_3d[:, 0])
    
    # Common valid points
    valid_both = valid_std & valid_agent
    n_common = np.sum(valid_both)
    n_total = len(std_recon_3d)
    
    print(f"[INFO] Total particles: {n_total}")
    print(f"[INFO] Standard valid: {np.sum(valid_std)}")
    print(f"[INFO] Agent valid: {np.sum(valid_agent)}")
    print(f"[INFO] Common valid: {n_common}")
    
    if n_common == 0:
        print("[ERROR] No common valid reconstructions!")
        sys.exit(1)
    
    # Compute metrics for agent (using std as ground truth)
    agent_metrics = evaluate_results(
        gt_3d=std_recon_3d,
        recon_3d=agent_recon_3d,
        n_views=agent_n_views,
        reproj_errors=agent_reproj_errors,
        cameras=cameras,
        config=config,
        results_dir=None
    )
    
    # Compute metrics for standard (self-comparison, should be perfect)
    std_metrics = evaluate_results(
        gt_3d=std_recon_3d,
        recon_3d=std_recon_3d,
        n_views=std_n_views,
        reproj_errors=std_reproj_errors,
        cameras=cameras,
        config=config,
        results_dir=None
    )
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Agent Metrics:")
    print(f"  - RMSE: {agent_metrics['rmse_3d_mm']:.6f} mm")
    print(f"  - Mean Error: {agent_metrics['mean_error_mm']:.6f} mm")
    print(f"  - Correlation Mean: {agent_metrics['correlation_mean']:.6f}")
    print(f"  - Success Rate: {agent_metrics['success_rate']:.4f}")
    print(f"  - PSNR: {agent_metrics['psnr_db']:.2f} dB")
    
    print(f"\nStandard Metrics (self-reference):")
    print(f"  - RMSE: {std_metrics['rmse_3d_mm']:.6f} mm")
    print(f"  - Correlation Mean: {std_metrics['correlation_mean']:.6f}")
    print(f"  - Success Rate: {std_metrics['success_rate']:.4f}")
    
    # Additional direct comparison
    std_valid_pts = std_recon_3d[valid_both]
    agent_valid_pts = agent_recon_3d[valid_both]
    
    direct_errors = np.linalg.norm(std_valid_pts - agent_valid_pts, axis=1)
    direct_rmse = np.sqrt(np.mean(direct_errors ** 2))
    direct_max_error = np.max(direct_errors)
    direct_mean_error = np.mean(direct_errors)
    
    print(f"\nDirect Comparison (Agent vs Standard):")
    print(f"  - Direct RMSE: {direct_rmse:.6f} mm")
    print(f"  - Direct Mean Error: {direct_mean_error:.6f} mm")
    print(f"  - Direct Max Error: {direct_max_error:.6f} mm")
    
    # Success criteria
    # For an inversion algorithm, we expect the agent to produce nearly identical results
    # Allow a small tolerance for numerical differences
    TOLERANCE_RMSE = 1e-3  # 1 micrometer tolerance
    TOLERANCE_SUCCESS_RATE = 0.01  # 1% tolerance on success rate
    
    success = True
    
    # Check if agent produces similar results to standard
    if direct_rmse > TOLERANCE_RMSE:
        # If RMSE is not near-zero, check if it's still reasonably small
        # relative to the reconstruction quality
        if direct_rmse > 0.1:  # More than 0.1mm difference is concerning
            print(f"\n[WARNING] Direct RMSE ({direct_rmse:.6f}) exceeds 0.1mm threshold")
            # Check correlation as fallback metric
            if agent_metrics['correlation_mean'] < 0.99:
                print(f"[FAIL] Correlation ({agent_metrics['correlation_mean']:.6f}) is below 0.99")
                success = False
    
    # Check success rate
    success_rate_diff = abs(agent_metrics['success_rate'] - std_metrics['success_rate'])
    if success_rate_diff > TOLERANCE_SUCCESS_RATE:
        print(f"\n[WARNING] Success rate difference ({success_rate_diff:.4f}) exceeds tolerance")
        if agent_metrics['success_rate'] < std_metrics['success_rate'] * 0.95:
            print(f"[FAIL] Agent success rate significantly lower than standard")
            success = False
    
    # Final verdict
    print("\n" + "="*60)
    if success:
        print("TEST PASSED: Agent reconstruction matches standard within tolerance")
        print("="*60)
        sys.exit(0)
    else:
        print("TEST FAILED: Agent reconstruction differs significantly from standard")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()