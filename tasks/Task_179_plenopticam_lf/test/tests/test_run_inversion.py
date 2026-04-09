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
from skimage.metrics import structural_similarity as ssim
import json

# Inject the referee function (evaluate_results)
def evaluate_results(gt_scene, gt_depth, recon_center, est_depth, raw_noisy, params, results_dir='results'):
    """
    Compute metrics, generate visualizations, and save results.
    
    Args:
        gt_scene: ground truth scene (2D array)
        gt_depth: ground truth depth map (2D array)
        recon_center: reconstructed center sub-aperture view (2D array)
        est_depth: estimated depth map (2D array)
        raw_noisy: noisy raw MLA image for visualization
        params: dictionary of parameters
        results_dir: directory to save results
    
    Returns:
        dict containing all computed metrics
    """
    os.makedirs(results_dir, exist_ok=True)

    # Compute PSNR for depth
    mse_depth = np.mean((gt_depth - est_depth) ** 2)
    if mse_depth < 1e-15:
        depth_psnr = 100.0
    else:
        data_range_depth = np.max(gt_depth) - np.min(gt_depth)
        if data_range_depth < 1e-15:
            data_range_depth = 1.0
        depth_psnr = 10.0 * np.log10(data_range_depth ** 2 / mse_depth)

    # Compute correlation coefficient for depth
    depth_cc = float(np.corrcoef(gt_depth.ravel(), est_depth.ravel())[0, 1])

    # Compute PSNR for sub-aperture
    mse_sa = np.mean((gt_scene - recon_center) ** 2)
    if mse_sa < 1e-15:
        sa_psnr = 100.0
    else:
        data_range_sa = np.max(gt_scene) - np.min(gt_scene)
        if data_range_sa < 1e-15:
            data_range_sa = 1.0
        sa_psnr = 10.0 * np.log10(data_range_sa ** 2 / mse_sa)

    # Compute SSIM for sub-aperture
    data_range_ssim = max(gt_scene.max() - gt_scene.min(), 
                          recon_center.max() - recon_center.min(), 1e-6)
    sa_ssim = float(ssim(gt_scene, recon_center, data_range=data_range_ssim))

    metrics = {
        "depth_psnr_dB": round(depth_psnr, 2),
        "depth_cc": round(depth_cc, 4),
        "subaperture_psnr_dB": round(sa_psnr, 2),
        "subaperture_ssim": round(sa_ssim, 4),
        "noise_std": params['noise_std'],
        "n_angular": params['n_angular'],
        "scene_size": params['scene_size'],
    }

    # Save metrics to JSON
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print metrics
    print(f"\n{'='*50}")
    print(f"  Depth  PSNR : {depth_psnr:.2f} dB")
    print(f"  Depth  CC   : {depth_cc:.4f}")
    print(f"  SA     PSNR : {sa_psnr:.2f} dB")
    print(f"  SA     SSIM : {sa_ssim:.4f}")
    print(f"{'='*50}\n")

    # Generate visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # (a) GT center sub-aperture
    ax = axes[0, 0]
    ax.imshow(gt_scene, cmap='gray', vmin=0, vmax=1)
    ax.set_title('(a) GT Center View')
    ax.axis('off')

    # (b) Raw MLA image (small patch)
    ax = axes[0, 1]
    n_angular = params['n_angular']
    patch_size = min(60 * n_angular, raw_noisy.shape[0])
    ax.imshow(raw_noisy[:patch_size, :patch_size], cmap='gray', vmin=0, vmax=1)
    ax.set_title('(b) Raw MLA Image (patch)')
    ax.axis('off')

    # (c) Reconstructed center sub-aperture
    ax = axes[0, 2]
    ax.imshow(recon_center, cmap='gray', vmin=0, vmax=1)
    ax.set_title('(c) Reconstructed Center View')
    ax.axis('off')

    # (d) GT depth map
    ax = axes[1, 0]
    vmin_d, vmax_d = gt_depth.min(), gt_depth.max()
    im_d = ax.imshow(gt_depth, cmap='viridis', vmin=vmin_d, vmax=vmax_d)
    ax.set_title('(d) GT Depth Map')
    ax.axis('off')
    plt.colorbar(im_d, ax=ax, fraction=0.046)

    # (e) Estimated depth map
    ax = axes[1, 1]
    im_e = ax.imshow(est_depth, cmap='viridis', vmin=vmin_d, vmax=vmax_d)
    ax.set_title('(e) Estimated Depth Map')
    ax.axis('off')
    plt.colorbar(im_e, ax=ax, fraction=0.046)

    # (f) Depth error map
    ax = axes[1, 2]
    err = np.abs(gt_depth - est_depth)
    im_f = ax.imshow(err, cmap='hot')
    ax.set_title('(f) Depth Error Map')
    ax.axis('off')
    plt.colorbar(im_f, ax=ax, fraction=0.046)

    plt.suptitle('Task 179: Light Field Reconstruction (plenopticam_lf)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'reconstruction_result.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Visualization saved to {save_path}")

    # Save arrays
    np.save(os.path.join(results_dir, 'ground_truth.npy'), gt_depth)
    np.save(os.path.join(results_dir, 'reconstruction.npy'), est_depth)
    print("[DONE] All results saved to results/")

    return metrics


def main():
    # Data paths
    data_paths = ['/data/yjh/plenopticam_lf_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Categorize files
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
    
    try:
        # Load primary (outer) data
        if not outer_files:
            print("[ERROR] No outer data file found!")
            sys.exit(1)
        
        outer_path = outer_files[0]
        print(f"[INFO] Loading outer data from: {outer_path}")
        
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"[INFO] Outer data keys: {outer_data.keys()}")
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"[INFO] Number of args: {len(args)}")
        print(f"[INFO] Kwargs keys: {kwargs.keys()}")
        
        # Run the agent's run_inversion
        print("[INFO] Running agent's run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        
        # Check if inner files exist (chained execution)
        if inner_files:
            # Chained execution pattern
            print("[INFO] Detected chained execution pattern")
            inner_path = inner_files[0]
            print(f"[INFO] Loading inner data from: {inner_path}")
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Execute the returned callable
            if callable(agent_output):
                final_result = agent_output(*inner_args, **inner_kwargs)
            else:
                final_result = agent_output
        else:
            # Direct execution pattern
            print("[INFO] Direct execution pattern")
            final_result = agent_output
            std_result = std_output
        
        # Extract results
        print(f"[INFO] Agent output type: {type(final_result)}")
        print(f"[INFO] Standard output type: {type(std_result)}")
        
        # The output should be a dict with keys: lf_recon, recon_center, est_depth
        if isinstance(final_result, dict):
            agent_recon_center = final_result.get('recon_center')
            agent_est_depth = final_result.get('est_depth')
            print(f"[INFO] Agent recon_center shape: {agent_recon_center.shape if agent_recon_center is not None else None}")
            print(f"[INFO] Agent est_depth shape: {agent_est_depth.shape if agent_est_depth is not None else None}")
        else:
            print(f"[ERROR] Unexpected agent output format: {type(final_result)}")
            sys.exit(1)
        
        if isinstance(std_result, dict):
            std_recon_center = std_result.get('recon_center')
            std_est_depth = std_result.get('est_depth')
            print(f"[INFO] Standard recon_center shape: {std_recon_center.shape if std_recon_center is not None else None}")
            print(f"[INFO] Standard est_depth shape: {std_est_depth.shape if std_est_depth is not None else None}")
        else:
            print(f"[ERROR] Unexpected standard output format: {type(std_result)}")
            sys.exit(1)
        
        # Extract input data for evaluation
        # args[0] is raw_noisy, args[1] or kwargs['params'] is params
        raw_noisy = args[0] if len(args) > 0 else kwargs.get('raw_noisy')
        params = args[1] if len(args) > 1 else kwargs.get('params', {})
        
        print(f"[INFO] raw_noisy shape: {raw_noisy.shape if raw_noisy is not None else None}")
        print(f"[INFO] params: {params}")
        
        # For evaluation, we need gt_scene and gt_depth
        # These should be in params or we can use std_result as ground truth
        gt_scene = params.get('gt_scene', std_recon_center)
        gt_depth = params.get('gt_depth', std_est_depth)
        
        # If gt_scene and gt_depth are not in params, we'll use the standard output as ground truth
        if gt_scene is None:
            gt_scene = std_recon_center
        if gt_depth is None:
            gt_depth = std_est_depth
        
        print(f"[INFO] gt_scene shape: {gt_scene.shape if gt_scene is not None else None}")
        print(f"[INFO] gt_depth shape: {gt_depth.shape if gt_depth is not None else None}")
        
        # Ensure params has required fields for evaluate_results
        if 'noise_std' not in params:
            params['noise_std'] = 0.0
        if 'n_angular' not in params:
            params['n_angular'] = 5
        if 'scene_size' not in params:
            params['scene_size'] = agent_recon_center.shape[0] if agent_recon_center is not None else 128
        
        # Evaluate agent's results
        print("\n[INFO] Evaluating Agent's results...")
        agent_metrics = evaluate_results(
            gt_scene=gt_scene,
            gt_depth=gt_depth,
            recon_center=agent_recon_center,
            est_depth=agent_est_depth,
            raw_noisy=raw_noisy,
            params=params,
            results_dir='results_agent'
        )
        
        # Evaluate standard results
        print("\n[INFO] Evaluating Standard results...")
        std_metrics = evaluate_results(
            gt_scene=gt_scene,
            gt_depth=gt_depth,
            recon_center=std_recon_center,
            est_depth=std_est_depth,
            raw_noisy=raw_noisy,
            params=params,
            results_dir='results_std'
        )
        
        # Compare metrics
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        print(f"\nAgent Metrics: {agent_metrics}")
        print(f"Standard Metrics: {std_metrics}")
        
        # Extract primary metrics for comparison
        agent_depth_psnr = agent_metrics.get('depth_psnr_dB', 0)
        std_depth_psnr = std_metrics.get('depth_psnr_dB', 0)
        
        agent_depth_cc = agent_metrics.get('depth_cc', 0)
        std_depth_cc = std_metrics.get('depth_cc', 0)
        
        agent_sa_psnr = agent_metrics.get('subaperture_psnr_dB', 0)
        std_sa_psnr = std_metrics.get('subaperture_psnr_dB', 0)
        
        agent_sa_ssim = agent_metrics.get('subaperture_ssim', 0)
        std_sa_ssim = std_metrics.get('subaperture_ssim', 0)
        
        print(f"\nScores -> Agent Depth PSNR: {agent_depth_psnr}, Standard Depth PSNR: {std_depth_psnr}")
        print(f"Scores -> Agent Depth CC: {agent_depth_cc}, Standard Depth CC: {std_depth_cc}")
        print(f"Scores -> Agent SA PSNR: {agent_sa_psnr}, Standard SA PSNR: {std_sa_psnr}")
        print(f"Scores -> Agent SA SSIM: {agent_sa_ssim}, Standard SA SSIM: {std_sa_ssim}")
        
        # Determine success (higher is better for PSNR, CC, SSIM)
        # Allow 10% margin
        margin = 0.9
        
        success = True
        
        # Check depth PSNR
        if agent_depth_psnr < std_depth_psnr * margin:
            print(f"[WARN] Agent Depth PSNR ({agent_depth_psnr}) significantly lower than Standard ({std_depth_psnr})")
            success = False
        
        # Check depth CC
        if agent_depth_cc < std_depth_cc * margin:
            print(f"[WARN] Agent Depth CC ({agent_depth_cc}) significantly lower than Standard ({std_depth_cc})")
            success = False
        
        # Check SA PSNR
        if agent_sa_psnr < std_sa_psnr * margin:
            print(f"[WARN] Agent SA PSNR ({agent_sa_psnr}) significantly lower than Standard ({std_sa_psnr})")
            success = False
        
        # Check SA SSIM
        if agent_sa_ssim < std_sa_ssim * margin:
            print(f"[WARN] Agent SA SSIM ({agent_sa_ssim}) significantly lower than Standard ({std_sa_ssim})")
            success = False
        
        if success:
            print("\n[SUCCESS] Agent performance is acceptable!")
            sys.exit(0)
        else:
            print("\n[FAILURE] Agent performance degraded significantly!")
            sys.exit(1)
            
    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()