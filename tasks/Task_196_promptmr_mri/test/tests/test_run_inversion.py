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
import json
from skimage.metrics import structural_similarity as ssim

# ============================================================
# Inject the Referee (evaluate_results) verbatim from Reference B
# ============================================================

def compute_psnr(gt, recon):
    """Compute PSNR."""
    mse = np.mean((gt - recon)**2)
    if mse < 1e-15:
        return 100.0
    data_range = gt.max() - gt.min()
    return 10.0 * np.log10(data_range**2 / mse)

def compute_ssim(gt, recon):
    """Compute SSIM."""
    data_range = gt.max() - gt.min()
    return ssim(gt, recon, data_range=data_range)

def evaluate_results(data_dict, result_dict, output_dir='results'):
    """
    Evaluate reconstruction quality and save results.
    """
    print("\n[4/4] Evaluating metrics...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    t1_gt = data_dict['t1_gt']
    t2_gt = data_dict['t2_gt']
    N = data_dict['N']
    t1_acceleration = data_dict['t1_acceleration']
    t2_acceleration = data_dict['t2_acceleration']
    
    recon_t1 = result_dict['recon_t1']
    recon_t2 = result_dict['recon_t2']
    zf_t1 = result_dict['zf_t1']
    zf_t2 = result_dict['zf_t2']
    n_iter = result_dict['n_iter']
    
    # Compute metrics for zero-filled
    psnr_zf_t1 = compute_psnr(t1_gt, zf_t1)
    psnr_zf_t2 = compute_psnr(t2_gt, zf_t2)
    ssim_zf_t1 = compute_ssim(t1_gt, zf_t1)
    ssim_zf_t2 = compute_ssim(t2_gt, zf_t2)
    
    # Compute metrics for reconstruction
    psnr_t1 = compute_psnr(t1_gt, recon_t1)
    psnr_t2 = compute_psnr(t2_gt, recon_t2)
    ssim_t1 = compute_ssim(t1_gt, recon_t1)
    ssim_t2 = compute_ssim(t2_gt, recon_t2)
    
    psnr_avg = (psnr_t1 + psnr_t2) / 2.0
    ssim_avg = (ssim_t1 + ssim_t2) / 2.0
    
    print(f"\n  Zero-filled baselines:")
    print(f"    T1: PSNR={psnr_zf_t1:.2f} dB, SSIM={ssim_zf_t1:.4f}")
    print(f"    T2: PSNR={psnr_zf_t2:.2f} dB, SSIM={ssim_zf_t2:.4f}")
    print(f"\n  CS-TV Reconstruction:")
    print(f"    T1 ({t1_acceleration}x): PSNR={psnr_t1:.2f} dB, SSIM={ssim_t1:.4f}")
    print(f"    T2 ({t2_acceleration}x): PSNR={psnr_t2:.2f} dB, SSIM={ssim_t2:.4f}")
    print(f"    Average: PSNR={psnr_avg:.2f} dB, SSIM={ssim_avg:.4f}")
    
    # Save metrics
    metrics = {
        "task": "promptmr_mri",
        "method": "FISTA-TV CS-MRI Reconstruction",
        "psnr_t1": round(psnr_t1, 2),
        "ssim_t1": round(ssim_t1, 4),
        "psnr_t2": round(psnr_t2, 2),
        "ssim_t2": round(ssim_t2, 4),
        "psnr_avg": round(psnr_avg, 2),
        "ssim_avg": round(ssim_avg, 4),
        "psnr_zf_t1": round(psnr_zf_t1, 2),
        "psnr_zf_t2": round(psnr_zf_t2, 2),
        "ssim_zf_t1": round(ssim_zf_t1, 4),
        "ssim_zf_t2": round(ssim_zf_t2, 4),
        "t1_acceleration": t1_acceleration,
        "t2_acceleration": t2_acceleration,
        "image_size": N,
        "fista_iterations": n_iter,
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to {output_dir}/metrics.json")
    
    # Save arrays
    gt_stack = np.stack([t1_gt, t2_gt], axis=0)
    recon_stack = np.stack([recon_t1, recon_t2], axis=0)
    np.save(os.path.join(output_dir, 'ground_truth.npy'), gt_stack)
    np.save(os.path.join(output_dir, 'reconstruction.npy'), recon_stack)
    print(f"  Arrays saved: ground_truth.npy {gt_stack.shape}, reconstruction.npy {recon_stack.shape}")
    
    # Visualization
    print("\n  Generating visualization...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    vmax_err = 0.15
    
    # Row 1: T1
    axes[0, 0].imshow(t1_gt, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('T1 Ground Truth', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(zf_t1, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'T1 Zero-filled ({t1_acceleration}x)\nPSNR={psnr_zf_t1:.1f}dB', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(recon_t1, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(f'T1 CS-TV Recon\nPSNR={psnr_t1:.1f}dB, SSIM={ssim_t1:.3f}', fontsize=12)
    axes[0, 2].axis('off')
    
    err_t1 = np.abs(t1_gt - recon_t1)
    im1 = axes[0, 3].imshow(err_t1, cmap='hot', vmin=0, vmax=vmax_err)
    axes[0, 3].set_title('T1 Error (×5)', fontsize=12)
    axes[0, 3].axis('off')
    plt.colorbar(im1, ax=axes[0, 3], fraction=0.046)
    
    # Row 2: T2
    axes[1, 0].imshow(t2_gt, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('T2 Ground Truth', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(zf_t2, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title(f'T2 Zero-filled ({t2_acceleration}x)\nPSNR={psnr_zf_t2:.1f}dB', fontsize=12)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(recon_t2, cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title(f'T2 CS-TV Recon\nPSNR={psnr_t2:.1f}dB, SSIM={ssim_t2:.3f}', fontsize=12)
    axes[1, 2].axis('off')
    
    err_t2 = np.abs(t2_gt - recon_t2)
    im2 = axes[1, 3].imshow(err_t2, cmap='hot', vmin=0, vmax=vmax_err)
    axes[1, 3].set_title('T2 Error (×5)', fontsize=12)
    axes[1, 3].axis('off')
    plt.colorbar(im2, ax=axes[1, 3], fraction=0.046)
    
    fig.suptitle(
        f'Multi-Contrast MRI Reconstruction (PromptMR-style)\n'
        f'T1({t1_acceleration}x): PSNR={psnr_t1:.2f}dB/SSIM={ssim_t1:.4f}  |  '
        f'T2({t2_acceleration}x): PSNR={psnr_t2:.2f}dB/SSIM={ssim_t2:.4f}  |  '
        f'Avg: PSNR={psnr_avg:.2f}dB/SSIM={ssim_avg:.4f}',
        fontsize=14, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(output_dir, 'reconstruction_result.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved to {output_dir}/reconstruction_result.png")
    
    print("\n" + "=" * 60)
    print(f"DONE — Average PSNR: {psnr_avg:.2f} dB, SSIM: {ssim_avg:.4f}")
    print("=" * 60)
    
    return metrics


# ============================================================
# Main Test Logic
# ============================================================

def main():
    data_paths = ['/data/yjh/promptmr_mri_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
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
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)
    
    func_name = outer_data.get('func_name', 'run_inversion')
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Function: {func_name}")
    print(f"Args count: {len(args)}, Kwargs keys: {list(kwargs.keys())}")
    
    # Run the agent's implementation
    print("\n--- Running agent's run_inversion ---")
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"ERROR: Agent run_inversion failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is chained execution
    if len(inner_paths) > 0:
        # Pattern 2: Chained execution
        print(f"\nDetected chained execution with {len(inner_paths)} inner file(s)")
        for ip in inner_paths:
            print(f"Loading inner data from: {ip}")
            with open(ip, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            try:
                final_result = agent_output(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Inner call failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Pattern 1: Direct execution
        print("\nDirect execution pattern detected.")
        final_result = agent_output
        std_result = std_output
    
    # Now we need to evaluate both results using evaluate_results
    # evaluate_results expects (data_dict, result_dict, output_dir)
    # data_dict is the input (args[0] which is data_dict)
    # result_dict is the output from run_inversion
    
    # Extract the data_dict (first argument to run_inversion)
    if len(args) > 0:
        data_dict = args[0]
    elif 'data_dict' in kwargs:
        data_dict = kwargs['data_dict']
    else:
        print("ERROR: Cannot find data_dict in inputs")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Evaluating AGENT output...")
    print("=" * 60)
    try:
        agent_metrics = evaluate_results(data_dict, final_result, output_dir='results_agent')
    except Exception as e:
        print(f"ERROR: evaluate_results failed on agent output: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Evaluating STANDARD output...")
    print("=" * 60)
    try:
        std_metrics = evaluate_results(data_dict, std_result, output_dir='results_std')
    except Exception as e:
        print(f"ERROR: evaluate_results failed on standard output: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract primary metrics for comparison
    agent_psnr_avg = agent_metrics['psnr_avg']
    std_psnr_avg = std_metrics['psnr_avg']
    agent_ssim_avg = agent_metrics['ssim_avg']
    std_ssim_avg = std_metrics['ssim_avg']
    
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  Agent  -> PSNR_avg: {agent_psnr_avg:.2f} dB, SSIM_avg: {agent_ssim_avg:.4f}")
    print(f"  Standard -> PSNR_avg: {std_psnr_avg:.2f} dB, SSIM_avg: {std_ssim_avg:.4f}")
    
    # Detailed per-contrast comparison
    print(f"\n  T1 PSNR: Agent={agent_metrics['psnr_t1']:.2f}, Std={std_metrics['psnr_t1']:.2f}")
    print(f"  T1 SSIM: Agent={agent_metrics['ssim_t1']:.4f}, Std={std_metrics['ssim_t1']:.4f}")
    print(f"  T2 PSNR: Agent={agent_metrics['psnr_t2']:.2f}, Std={std_metrics['psnr_t2']:.2f}")
    print(f"  T2 SSIM: Agent={agent_metrics['ssim_t2']:.4f}, Std={std_metrics['ssim_t2']:.4f}")
    
    # Verification: Higher is better for both PSNR and SSIM
    # Allow 10% margin for PSNR and 5% margin for SSIM
    psnr_threshold = std_psnr_avg * 0.90  # 10% margin
    ssim_threshold = std_ssim_avg * 0.95   # 5% margin
    
    psnr_pass = agent_psnr_avg >= psnr_threshold
    ssim_pass = agent_ssim_avg >= ssim_threshold
    
    print(f"\n  PSNR check: {agent_psnr_avg:.2f} >= {psnr_threshold:.2f} (90% of std)? {'PASS' if psnr_pass else 'FAIL'}")
    print(f"  SSIM check: {agent_ssim_avg:.4f} >= {ssim_threshold:.4f} (95% of std)? {'PASS' if ssim_pass else 'FAIL'}")
    
    if psnr_pass and ssim_pass:
        print("\n*** TEST PASSED: Agent performance is within acceptable range. ***")
        sys.exit(0)
    else:
        print("\n*** TEST FAILED: Agent performance degraded significantly. ***")
        sys.exit(1)


if __name__ == '__main__':
    main()