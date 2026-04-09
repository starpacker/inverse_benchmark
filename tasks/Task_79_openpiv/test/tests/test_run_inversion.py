import sys
import os
import dill
import numpy as np
import traceback

# Import target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from skimage.metrics import structural_similarity as ssim

# Inject the referee function (evaluate_results)
def evaluate_results(gt_u, gt_v, recon_u, recon_v, frame_a, frame_b,
                     x_grid, y_grid, results_dir):
    """
    Compute PIV-specific evaluation metrics and generate visualizations.
    
    Args:
        gt_u: Ground truth u velocity field
        gt_v: Ground truth v velocity field
        recon_u: Reconstructed u velocity field
        recon_v: Reconstructed v velocity field
        frame_a: First frame
        frame_b: Second frame
        x_grid: x coordinates of PIV grid
        y_grid: y coordinates of PIV grid
        results_dir: Directory to save results
    
    Returns:
        dict of metrics including:
        - rmse_u, rmse_v: RMSE for each component
        - rmse_magnitude: RMSE for velocity magnitude
        - cc_u, cc_v: Correlation coefficients
        - re: Relative error
        - psnr: PSNR based on velocity magnitude range
        - aee: Average Endpoint Error
        - ssim: Structural similarity
    """
    # Velocity magnitudes
    gt_mag = np.sqrt(gt_u**2 + gt_v**2)
    recon_mag = np.sqrt(recon_u**2 + recon_v**2)
    error_mag = np.abs(gt_mag - recon_mag)
    
    # RMSE per component
    rmse_u = np.sqrt(np.mean((gt_u - recon_u)**2))
    rmse_v = np.sqrt(np.mean((gt_v - recon_v)**2))
    
    # RMSE of magnitude
    rmse_mag = np.sqrt(np.mean((gt_mag - recon_mag)**2))
    
    # Correlation coefficient
    cc_u = np.corrcoef(gt_u.flatten(), recon_u.flatten())[0, 1]
    cc_v = np.corrcoef(gt_v.flatten(), recon_v.flatten())[0, 1]
    cc_mag = np.corrcoef(gt_mag.flatten(), recon_mag.flatten())[0, 1]
    
    # Relative Error
    gt_norm = np.sqrt(np.mean(gt_u**2 + gt_v**2))
    error_norm = np.sqrt(np.mean((gt_u - recon_u)**2 + (gt_v - recon_v)**2))
    re = error_norm / gt_norm if gt_norm > 0 else float('inf')
    
    # Average Endpoint Error
    aee = np.mean(np.sqrt((gt_u - recon_u)**2 + (gt_v - recon_v)**2))
    
    # PSNR based on velocity magnitude
    data_range = gt_mag.max() - gt_mag.min()
    mse = np.mean((gt_mag - recon_mag)**2)
    psnr = 10 * np.log10(data_range**2 / mse) if mse > 0 else float('inf')
    
    # SSIM of velocity magnitude
    gt_mag_norm = (gt_mag - gt_mag.min()) / (gt_mag.max() - gt_mag.min() + 1e-10)
    recon_mag_norm = (recon_mag - recon_mag.min()) / (recon_mag.max() - recon_mag.min() + 1e-10)
    min_dim = min(gt_mag_norm.shape)
    win_size = min(7, min_dim) if min_dim >= 3 else 3
    if win_size % 2 == 0:
        win_size -= 1
    ssim_val = ssim(gt_mag_norm, recon_mag_norm, data_range=1.0, win_size=win_size)
    
    metrics = {
        'rmse_u': float(rmse_u),
        'rmse_v': float(rmse_v),
        'rmse_magnitude': float(rmse_mag),
        'cc_u': float(cc_u),
        'cc_v': float(cc_v),
        'cc_magnitude': float(cc_mag),
        'relative_error': float(re),
        'average_endpoint_error': float(aee),
        'psnr': float(psnr),
        'ssim': float(ssim_val),
        'rmse': float(rmse_mag),
    }
    
    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")
    
    # Save arrays
    gt_velocity = np.stack([gt_u, gt_v], axis=0)
    recon_velocity = np.stack([recon_u, recon_v], axis=0)
    input_data = np.stack([frame_a, frame_b], axis=0)
    
    np.save(os.path.join(results_dir, "ground_truth.npy"), gt_velocity)
    np.save(os.path.join(results_dir, "reconstruction.npy"), recon_velocity)
    np.save(os.path.join(results_dir, "input.npy"), input_data)
    print(f"[SAVE] Ground truth shape: {gt_velocity.shape} → ground_truth.npy")
    print(f"[SAVE] Reconstruction shape: {recon_velocity.shape} → reconstruction.npy")
    print(f"[SAVE] Input shape: {input_data.shape} → input.npy")
    
    # Generate visualization
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    
    # Row 1: Images and velocity fields
    axes[0, 0].imshow(frame_a, cmap='gray')
    axes[0, 0].set_title('Frame A (Particle Image)', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(frame_b, cmap='gray')
    axes[0, 1].set_title('Frame B (Displaced Particles)', fontsize=12)
    axes[0, 1].axis('off')
    
    im_gt = axes[0, 2].imshow(gt_mag, cmap='jet', origin='upper')
    axes[0, 2].set_title('Ground Truth |V|', fontsize=12)
    plt.colorbar(im_gt, ax=axes[0, 2], fraction=0.046, label='pixels/frame')
    
    im_recon = axes[0, 3].imshow(recon_mag, cmap='jet', origin='upper',
                                  vmin=gt_mag.min(), vmax=gt_mag.max())
    axes[0, 3].set_title('Reconstructed |V|', fontsize=12)
    plt.colorbar(im_recon, ax=axes[0, 3], fraction=0.046, label='pixels/frame')
    
    # Row 2: Quiver plots and error
    skip = 1
    axes[1, 0].quiver(x_grid[::skip, ::skip], y_grid[::skip, ::skip],
                       gt_u[::skip, ::skip], -gt_v[::skip, ::skip],
                       gt_mag[::skip, ::skip], cmap='jet',
                       scale=None, width=0.004)
    axes[1, 0].set_title('GT Velocity Field', fontsize=12)
    axes[1, 0].set_aspect('equal')
    axes[1, 0].invert_yaxis()
    
    axes[1, 1].quiver(x_grid[::skip, ::skip], y_grid[::skip, ::skip],
                       recon_u[::skip, ::skip], -recon_v[::skip, ::skip],
                       recon_mag[::skip, ::skip], cmap='jet',
                       scale=None, width=0.004)
    axes[1, 1].set_title('Reconstructed Velocity Field', fontsize=12)
    axes[1, 1].set_aspect('equal')
    axes[1, 1].invert_yaxis()
    
    im_err = axes[1, 2].imshow(error_mag, cmap='hot', origin='upper')
    axes[1, 2].set_title('Velocity Magnitude Error', fontsize=12)
    plt.colorbar(im_err, ax=axes[1, 2], fraction=0.046, label='pixels/frame')
    
    axes[1, 3].scatter(gt_mag.flatten(), recon_mag.flatten(), alpha=0.5, s=10, c='steelblue')
    max_val = max(gt_mag.max(), recon_mag.max()) * 1.1
    axes[1, 3].plot([0, max_val], [0, max_val], 'r--', lw=2, label='Identity')
    axes[1, 3].set_xlabel('GT |V| (px/frame)', fontsize=11)
    axes[1, 3].set_ylabel('Recon |V| (px/frame)', fontsize=11)
    axes[1, 3].set_title(f'GT vs Recon (CC={metrics["cc_magnitude"]:.4f})', fontsize=12)
    axes[1, 3].legend()
    axes[1, 3].set_aspect('equal')
    axes[1, 3].set_xlim([0, max_val])
    axes[1, 3].set_ylim([0, max_val])
    
    fig.suptitle(
        f"OpenPIV — PIV Flow Reconstruction\n"
        f"PSNR={metrics['psnr']:.2f} dB | SSIM={metrics['ssim']:.4f} | "
        f"AEE={metrics['average_endpoint_error']:.4f} px | "
        f"CC={metrics['cc_magnitude']:.4f} | RE={metrics['relative_error']:.4f}",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {vis_path}")
    
    return metrics


def main():
    # Data paths provided
    data_paths = ['/data/yjh/openpiv_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_files = []
    inner_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_files.append(path)
        else:
            outer_files.append(path)
    
    print(f"[INFO] Outer files: {outer_files}")
    print(f"[INFO] Inner files: {inner_files}")
    
    # We expect at least one outer file
    if not outer_files:
        print("[ERROR] No outer data file found!")
        sys.exit(1)
    
    # Load outer data
    outer_path = outer_files[0]
    print(f"[INFO] Loading outer data from: {outer_path}")
    
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print(f"[INFO] Outer data keys: {outer_data.keys()}")
    
    # Extract args and kwargs
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"[INFO] Args length: {len(args)}")
    print(f"[INFO] Kwargs: {list(kwargs.keys())}")
    
    # Run agent function
    print("\n[INFO] Running agent run_inversion...")
    try:
        agent_output = run_inversion(*args, **kwargs)
        print("[INFO] Agent function executed successfully")
    except Exception as e:
        print(f"[ERROR] Agent function failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check for chained execution (inner data)
    if inner_files:
        print("\n[INFO] Chained execution detected - processing inner data...")
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
        
        # Execute operator (agent_output should be callable)
        if callable(agent_output):
            try:
                final_result = agent_output(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"[ERROR] Inner function execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
        else:
            print("[ERROR] Agent output is not callable for chained execution")
            sys.exit(1)
    else:
        # Direct execution
        final_result = agent_output
        std_result = std_output
    
    print(f"\n[INFO] Final result type: {type(final_result)}")
    print(f"[INFO] Standard result type: {type(std_result)}")
    
    # Extract velocity fields from results
    if isinstance(final_result, dict):
        agent_u = final_result.get('u_recon', None)
        agent_v = final_result.get('v_recon', None)
        agent_x_grid = final_result.get('x_grid', None)
        agent_y_grid = final_result.get('y_grid', None)
    else:
        print("[ERROR] Agent result is not a dictionary")
        sys.exit(1)
    
    if isinstance(std_result, dict):
        std_u = std_result.get('u_recon', None)
        std_v = std_result.get('v_recon', None)
        std_x_grid = std_result.get('x_grid', None)
        std_y_grid = std_result.get('y_grid', None)
    else:
        print("[ERROR] Standard result is not a dictionary")
        sys.exit(1)
    
    # Get frame_a and frame_b from original args
    frame_a = args[0] if len(args) > 0 else kwargs.get('frame_a', None)
    frame_b = args[1] if len(args) > 1 else kwargs.get('frame_b', None)
    
    if frame_a is None or frame_b is None:
        print("[ERROR] Could not extract frame_a or frame_b from input args")
        sys.exit(1)
    
    # Create results directories
    agent_results_dir = './results_agent'
    std_results_dir = './results_std'
    
    # Use standard output as ground truth for evaluation
    # Evaluate agent results against standard results
    print("\n[INFO] Evaluating agent results...")
    try:
        agent_metrics = evaluate_results(
            gt_u=std_u,
            gt_v=std_v,
            recon_u=agent_u,
            recon_v=agent_v,
            frame_a=frame_a,
            frame_b=frame_b,
            x_grid=agent_x_grid,
            y_grid=agent_y_grid,
            results_dir=agent_results_dir
        )
        print(f"[INFO] Agent metrics: {agent_metrics}")
    except Exception as e:
        print(f"[ERROR] Agent evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Evaluate standard results against themselves (perfect score baseline)
    print("\n[INFO] Evaluating standard results (baseline)...")
    try:
        std_metrics = evaluate_results(
            gt_u=std_u,
            gt_v=std_v,
            recon_u=std_u,
            recon_v=std_v,
            frame_a=frame_a,
            frame_b=frame_b,
            x_grid=std_x_grid,
            y_grid=std_y_grid,
            results_dir=std_results_dir
        )
        print(f"[INFO] Standard metrics (perfect baseline): {std_metrics}")
    except Exception as e:
        print(f"[ERROR] Standard evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Compare metrics
    # Key metrics to compare: PSNR (higher is better), SSIM (higher is better), 
    # RMSE (lower is better), correlation (higher is better)
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    # Extract key metrics
    agent_psnr = agent_metrics.get('psnr', 0)
    agent_ssim = agent_metrics.get('ssim', 0)
    agent_rmse = agent_metrics.get('rmse_magnitude', float('inf'))
    agent_cc = agent_metrics.get('cc_magnitude', 0)
    agent_aee = agent_metrics.get('average_endpoint_error', float('inf'))
    
    print(f"\nAgent Performance:")
    print(f"  PSNR: {agent_psnr:.4f} dB")
    print(f"  SSIM: {agent_ssim:.4f}")
    print(f"  RMSE: {agent_rmse:.4f}")
    print(f"  CC:   {agent_cc:.4f}")
    print(f"  AEE:  {agent_aee:.4f}")
    
    # For a perfect match (agent == standard), we expect:
    # - PSNR = inf or very high
    # - SSIM = 1.0
    # - RMSE = 0
    # - CC = 1.0
    
    # Define acceptable thresholds
    # Since we're comparing agent to standard, we want them to be very similar
    psnr_threshold = 30.0  # PSNR should be high (> 30 dB indicates good match)
    ssim_threshold = 0.95  # SSIM should be close to 1.0
    rmse_threshold = 1.0   # RMSE should be low (depends on velocity scale)
    cc_threshold = 0.95    # Correlation should be high
    
    # Check if results are acceptable
    test_passed = True
    
    # Handle infinite PSNR (perfect match)
    if np.isinf(agent_psnr):
        print(f"\n[PASS] PSNR: Perfect match (inf)")
    elif agent_psnr >= psnr_threshold:
        print(f"\n[PASS] PSNR: {agent_psnr:.4f} >= {psnr_threshold}")
    else:
        print(f"\n[WARN] PSNR: {agent_psnr:.4f} < {psnr_threshold}")
        # Don't fail on PSNR alone if other metrics are good
    
    if agent_ssim >= ssim_threshold:
        print(f"[PASS] SSIM: {agent_ssim:.4f} >= {ssim_threshold}")
    else:
        print(f"[FAIL] SSIM: {agent_ssim:.4f} < {ssim_threshold}")
        test_passed = False
    
    if agent_rmse <= rmse_threshold:
        print(f"[PASS] RMSE: {agent_rmse:.4f} <= {rmse_threshold}")
    else:
        print(f"[WARN] RMSE: {agent_rmse:.4f} > {rmse_threshold}")
        # Check relative to the velocity scale
        max_velocity = max(np.max(np.abs(std_u)), np.max(np.abs(std_v)))
        relative_rmse = agent_rmse / (max_velocity + 1e-10)
        if relative_rmse > 0.1:  # More than 10% relative error
            print(f"[FAIL] Relative RMSE: {relative_rmse:.4f} > 0.1")
            test_passed = False
        else:
            print(f"[PASS] Relative RMSE: {relative_rmse:.4f} <= 0.1")
    
    if agent_cc >= cc_threshold:
        print(f"[PASS] CC: {agent_cc:.4f} >= {cc_threshold}")
    else:
        print(f"[FAIL] CC: {agent_cc:.4f} < {cc_threshold}")
        test_passed = False
    
    print("\n" + "="*60)
    
    if test_passed:
        print("[SUCCESS] Agent performance is acceptable!")
        print("="*60)
        sys.exit(0)
    else:
        print("[FAILURE] Agent performance degraded significantly!")
        print("="*60)
        sys.exit(1)


if __name__ == '__main__':
    main()