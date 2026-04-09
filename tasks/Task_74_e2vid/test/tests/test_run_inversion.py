import sys
import os
import dill
import numpy as np
import traceback
import json

# Import the target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_fn

# Inject the evaluate_results function (Reference B)
def evaluate_results(gt_frames, rec_frames, events, results_dir, working_dir):
    """
    Compute metrics and save results.
    
    Args:
        gt_frames: Ground truth frames
        rec_frames: Reconstructed frames
        events: List of events
        results_dir: Directory to save results
        working_dir: Working directory
    
    Returns:
        metrics: Dictionary of computed metrics
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Compute metrics
    n = min(len(gt_frames), len(rec_frames))
    psnr_list, ssim_list, cc_list = [], [], []

    for i in range(n):
        gt = gt_frames[i]
        rec = rec_frames[i]
        # Min-max normalization to [0, 1]
        gt_n = (gt - gt.min()) / (gt.max() - gt.min() + 1e-10)
        rec_n = (rec - rec.min()) / (rec.max() - rec.min() + 1e-10)

        data_range = 1.0
        mse = np.mean((gt_n - rec_n)**2)
        psnr_list.append(10 * np.log10(data_range**2 / max(mse, 1e-30)))
        ssim_list.append(ssim_fn(gt_n, rec_n, data_range=data_range))
        cc_list.append(np.corrcoef(gt_n.ravel(), rec_n.ravel())[0, 1])

    gt_all = gt_frames[:n].ravel()
    rec_all = rec_frames[:n].ravel()
    gt_all_n = (gt_all - gt_all.min()) / (gt_all.max() - gt_all.min() + 1e-10)
    rec_all_n = (rec_all - rec_all.min()) / (rec_all.max() - rec_all.min() + 1e-10)
    re = float(np.linalg.norm(gt_all_n - rec_all_n) /
               max(np.linalg.norm(gt_all_n), 1e-12))
    rmse = float(np.sqrt(np.mean((gt_all_n - rec_all_n)**2)))

    metrics = {
        "PSNR": float(np.mean(psnr_list)),
        "SSIM": float(np.mean(ssim_list)),
        "CC": float(np.mean(cc_list)),
        "RE": re,
        "RMSE": rmse,
    }
    
    # Save metrics
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save arrays
    np.save(os.path.join(results_dir, "reconstruction.npy"), rec_frames)
    np.save(os.path.join(results_dir, "ground_truth.npy"), gt_frames)
    np.save(os.path.join(working_dir, "recon_output.npy"), rec_frames)
    np.save(os.path.join(working_dir, "gt_output.npy"), gt_frames)
    
    # Visualization
    n_show = min(4, len(gt_frames))
    fig, axes = plt.subplots(3, n_show, figsize=(4 * n_show, 10))

    indices = np.linspace(0, len(gt_frames) - 1, n_show, dtype=int)

    for col, idx in enumerate(indices):
        gt = gt_frames[idx] / max(gt_frames[idx].max(), 1e-12)
        rec = rec_frames[idx] / max(rec_frames[idx].max(), 1e-12)

        axes[0, col].imshow(gt, cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(f'GT Frame {idx}')
        axes[0, col].axis('off')

        axes[1, col].imshow(rec, cmap='gray', vmin=0, vmax=1)
        axes[1, col].set_title(f'Recon Frame {idx}')
        axes[1, col].axis('off')

        axes[2, col].imshow(np.abs(gt - rec), cmap='hot', vmin=0, vmax=0.5)
        axes[2, col].set_title('|Error|')
        axes[2, col].axis('off')

    fig.suptitle(
        f"e2vid — Event Camera Reconstruction\n"
        f"Events: {len(events)} | PSNR={metrics['PSNR']:.1f} dB | "
        f"SSIM={metrics['SSIM']:.4f} | CC={metrics['CC']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(results_dir, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics


def main():
    # Data paths provided
    data_paths = ['/data/yjh/e2vid_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_data_file = None
    inner_data_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_files.append(path)
        else:
            outer_data_file = path
    
    if outer_data_file is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)
    
    # Load outer data
    print(f"Loading outer data from: {outer_data_file}")
    try:
        with open(outer_data_file, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"Number of args: {len(args)}")
    print(f"Kwargs keys: {list(kwargs.keys())}")
    
    # Execute the agent's run_inversion
    print("Running agent's run_inversion...")
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"ERROR: Agent function failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if we have inner data (chained execution)
    if inner_data_files:
        # Chained execution pattern
        print(f"Detected chained execution with {len(inner_data_files)} inner files")
        
        # Load inner data
        inner_data_file = inner_data_files[0]
        print(f"Loading inner data from: {inner_data_file}")
        try:
            with open(inner_data_file, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        # Execute chained call
        if callable(agent_output):
            print("Executing chained call...")
            try:
                final_agent_result = agent_output(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Chained call failed: {e}")
                traceback.print_exc()
                sys.exit(1)
        else:
            final_agent_result = agent_output
    else:
        # Direct execution pattern
        print("Direct execution pattern detected")
        final_agent_result = agent_output
        std_result = std_output
    
    # Extract reconstruction results
    # run_inversion returns: (best_rec, best_name, all_reconstructions)
    if isinstance(final_agent_result, tuple) and len(final_agent_result) >= 1:
        agent_rec = final_agent_result[0]  # best_rec
        agent_name = final_agent_result[1] if len(final_agent_result) > 1 else "Unknown"
        print(f"Agent selected method: {agent_name}")
    else:
        agent_rec = final_agent_result
        agent_name = "Unknown"
    
    if isinstance(std_result, tuple) and len(std_result) >= 1:
        std_rec = std_result[0]  # best_rec
        std_name = std_result[1] if len(std_result) > 1 else "Unknown"
        print(f"Standard selected method: {std_name}")
    else:
        std_rec = std_result
        std_name = "Unknown"
    
    # Extract gt_frames and events from the input args for evaluation
    # Based on the function signature:
    # run_inversion(events, gt_frames, aps_frames, height, width, n_output_frames, contrast_threshold, t_total, fps)
    events = args[0] if len(args) > 0 else kwargs.get('events', [])
    gt_frames = args[1] if len(args) > 1 else kwargs.get('gt_frames', None)
    
    if gt_frames is None:
        print("ERROR: Could not extract gt_frames from inputs")
        sys.exit(1)
    
    # Ensure gt_frames is numpy array
    if not isinstance(gt_frames, np.ndarray):
        gt_frames = np.array(gt_frames)
    
    # Ensure reconstructions are numpy arrays
    if not isinstance(agent_rec, np.ndarray):
        agent_rec = np.array(agent_rec)
    if not isinstance(std_rec, np.ndarray):
        std_rec = np.array(std_rec)
    
    print(f"GT frames shape: {gt_frames.shape}")
    print(f"Agent reconstruction shape: {agent_rec.shape}")
    print(f"Standard reconstruction shape: {std_rec.shape}")
    
    # Create directories for evaluation output
    working_dir = os.path.dirname(outer_data_file)
    if not working_dir:
        working_dir = '.'
    
    agent_results_dir = os.path.join(working_dir, 'agent_results')
    std_results_dir = os.path.join(working_dir, 'std_results')
    
    os.makedirs(agent_results_dir, exist_ok=True)
    os.makedirs(std_results_dir, exist_ok=True)
    
    # Evaluate agent's result
    print("Evaluating agent's reconstruction...")
    try:
        metrics_agent = evaluate_results(gt_frames, agent_rec, events, agent_results_dir, working_dir)
    except Exception as e:
        print(f"ERROR: Agent evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Evaluate standard result
    print("Evaluating standard reconstruction...")
    try:
        metrics_std = evaluate_results(gt_frames, std_rec, events, std_results_dir, working_dir)
    except Exception as e:
        print(f"ERROR: Standard evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract primary metrics (PSNR is higher-is-better)
    psnr_agent = metrics_agent.get('PSNR', 0)
    psnr_std = metrics_std.get('PSNR', 0)
    
    ssim_agent = metrics_agent.get('SSIM', 0)
    ssim_std = metrics_std.get('SSIM', 0)
    
    rmse_agent = metrics_agent.get('RMSE', float('inf'))
    rmse_std = metrics_std.get('RMSE', float('inf'))
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Agent Metrics:    PSNR={psnr_agent:.2f} dB, SSIM={ssim_agent:.4f}, RMSE={rmse_agent:.4f}")
    print(f"Standard Metrics: PSNR={psnr_std:.2f} dB, SSIM={ssim_std:.4f}, RMSE={rmse_std:.4f}")
    print("="*60)
    
    # Determine success based on PSNR (higher is better)
    # Allow 10% margin of error
    margin = 0.9
    
    psnr_pass = psnr_agent >= psnr_std * margin or psnr_agent >= psnr_std - 2.0  # Allow 2dB degradation
    ssim_pass = ssim_agent >= ssim_std * margin
    
    print(f"\nPSNR Check: Agent={psnr_agent:.2f}, Threshold={psnr_std * margin:.2f} -> {'PASS' if psnr_pass else 'FAIL'}")
    print(f"SSIM Check: Agent={ssim_agent:.4f}, Threshold={ssim_std * margin:.4f} -> {'PASS' if ssim_pass else 'FAIL'}")
    
    if psnr_pass and ssim_pass:
        print("\n✓ Performance verification PASSED")
        sys.exit(0)
    else:
        print("\n✗ Performance verification FAILED")
        print("Agent's reconstruction quality is significantly worse than standard.")
        sys.exit(1)


if __name__ == '__main__':
    main()