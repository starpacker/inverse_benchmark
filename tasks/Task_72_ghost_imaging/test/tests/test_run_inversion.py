import sys
import os
import dill
import numpy as np
import traceback
import json

# Import matplotlib with Agg backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import skimage for SSIM
from skimage.metrics import structural_similarity as ssim_fn

# Import the agent's function
from agent_run_inversion import run_inversion

# Inject the referee function verbatim
def evaluate_results(data, reconstruction_results, results_dir):
    """
    Compute metrics and visualize reconstruction results.
    
    Args:
        data: dict from load_and_preprocess_data
        reconstruction_results: dict or list of dicts from run_inversion
        results_dir: directory to save results
    
    Returns:
        dict containing metrics for the best reconstruction
    """
    os.makedirs(results_dir, exist_ok=True)
    
    img_gt = data['img_gt']
    config = data['config']
    compression_ratio = config['compression_ratio']
    
    def compute_metrics(gt, rec):
        gt_n = gt / max(gt.max(), 1e-12)
        rec_n = rec / max(rec.max(), 1e-12)
        data_range = 1.0
        mse = np.mean((gt_n - rec_n)**2)
        psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
        ssim_val = float(ssim_fn(gt_n, rec_n, data_range=data_range))
        cc = float(np.corrcoef(gt_n.ravel(), rec_n.ravel())[0, 1])
        re = float(np.linalg.norm(gt_n - rec_n) / max(np.linalg.norm(gt_n), 1e-12))
        rmse = float(np.sqrt(mse))
        return {"PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse}
    
    # Handle single result or list of results
    if isinstance(reconstruction_results, dict):
        reconstruction_results = [reconstruction_results]
    
    all_metrics = []
    for res in reconstruction_results:
        m = compute_metrics(img_gt, res['img_rec'])
        m['method'] = res['method']
        all_metrics.append(m)
        print(f"  {res['method']}: CC={m['CC']:.4f}, PSNR={m['PSNR']:.1f}")
    
    # Select best by CC
    best_idx = np.argmax([m['CC'] for m in all_metrics])
    best_result = reconstruction_results[best_idx]
    best_metrics = {k: v for k, v in all_metrics[best_idx].items() if k != 'method'}
    best_method = all_metrics[best_idx]['method']
    
    print(f"\n  → Using {best_method} (highest CC)")
    print("\n[Evaluation Metrics]:")
    for k, v in sorted(best_metrics.items()):
        print(f"  {k:20s} = {v}")
    
    # Visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(img_gt, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth')
    
    # Find correlation and fista results for display
    img_corr = None
    img_fista = None
    for res in reconstruction_results:
        if res['method'] == 'correlation':
            img_corr = res['img_rec']
        elif res['method'] == 'fista':
            img_fista = res['img_rec']
    
    if img_corr is not None:
        rec_corr_n = img_corr / max(img_corr.max(), 1e-12)
        axes[1].imshow(rec_corr_n, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Correlation GI')
    else:
        axes[1].set_title('Correlation GI (N/A)')
    
    if img_fista is not None:
        rec_fista_n = img_fista / max(img_fista.max(), 1e-12)
        axes[2].imshow(rec_fista_n, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('FISTA CS')
        err = np.abs(img_gt - rec_fista_n)
    else:
        rec_best = best_result['img_rec']
        rec_best_n = rec_best / max(rec_best.max(), 1e-12)
        axes[2].imshow(rec_best_n, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title(f'{best_method}')
        err = np.abs(img_gt - rec_best_n)
    
    axes[3].imshow(err, cmap='hot', vmin=0)
    axes[3].set_title('|Error|')
    
    for ax in axes:
        ax.axis('off')
    
    fig.suptitle(
        f"Ghost Imaging — Compressive Single-Pixel Reconstruction\n"
        f"M/N={compression_ratio:.0%} | PSNR={best_metrics['PSNR']:.1f} dB | "
        f"SSIM={best_metrics['SSIM']:.4f} | CC={best_metrics['CC']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(results_dir, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(best_metrics, f, indent=2)
    np.save(os.path.join(results_dir, "reconstruction.npy"), best_result['img_rec'])
    np.save(os.path.join(results_dir, "ground_truth.npy"), img_gt)
    
    return best_metrics


def main():
    # Data paths provided
    data_paths = ['/data/yjh/ghost_imaging_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_data_files = []
    inner_data_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_files.append(path)
        else:
            outer_data_files.append(path)
    
    print(f"[INFO] Outer data files: {outer_data_files}")
    print(f"[INFO] Inner data files: {inner_data_files}")
    
    # Determine execution pattern
    is_chained = len(inner_data_files) > 0
    
    try:
        # Load primary (outer) data
        if not outer_data_files:
            print("[ERROR] No outer data file found.")
            sys.exit(1)
        
        outer_path = outer_data_files[0]
        print(f"[INFO] Loading outer data from: {outer_path}")
        
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"[INFO] Running agent's run_inversion with args shape/types...")
        
        # Run the agent's function
        agent_output = run_inversion(*args, **kwargs)
        
        if is_chained:
            # Pattern 2: Chained Execution
            inner_path = inner_data_files[0]
            print(f"[INFO] Loading inner data from: {inner_path}")
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Execute the operator returned by run_inversion
            print("[INFO] Executing chained operator...")
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Pattern 1: Direct Execution
            final_result = agent_output
            std_result = std_output
        
        # Now we need to evaluate both results
        # The evaluate_results function expects 'data' (containing img_gt, config)
        # and 'reconstruction_results' (the output from run_inversion)
        
        # Extract the original data dict from args (first argument to run_inversion)
        if len(args) > 0:
            original_data = args[0]
        else:
            original_data = kwargs.get('data', None)
        
        if original_data is None:
            print("[ERROR] Could not find original data dict for evaluation.")
            sys.exit(1)
        
        # Create results directories
        agent_results_dir = './agent_results'
        std_results_dir = './std_results'
        
        print("\n[INFO] Evaluating agent's reconstruction...")
        agent_metrics = evaluate_results(original_data, final_result, agent_results_dir)
        
        print("\n[INFO] Evaluating standard reconstruction...")
        std_metrics = evaluate_results(original_data, std_result, std_results_dir)
        
        # Extract primary metrics for comparison
        # Using CC (correlation coefficient) as primary metric (higher is better)
        agent_cc = agent_metrics.get('CC', 0.0)
        std_cc = std_metrics.get('CC', 0.0)
        
        agent_psnr = agent_metrics.get('PSNR', 0.0)
        std_psnr = std_metrics.get('PSNR', 0.0)
        
        agent_ssim = agent_metrics.get('SSIM', 0.0)
        std_ssim = std_metrics.get('SSIM', 0.0)
        
        print(f"\n{'='*60}")
        print(f"[COMPARISON RESULTS]")
        print(f"{'='*60}")
        print(f"Metric          | Agent         | Standard      | Diff")
        print(f"{'-'*60}")
        print(f"CC              | {agent_cc:.6f}      | {std_cc:.6f}      | {agent_cc - std_cc:+.6f}")
        print(f"PSNR (dB)       | {agent_psnr:.2f}        | {std_psnr:.2f}        | {agent_psnr - std_psnr:+.2f}")
        print(f"SSIM            | {agent_ssim:.6f}      | {std_ssim:.6f}      | {agent_ssim - std_ssim:+.6f}")
        print(f"{'='*60}")
        
        # Determine success
        # For CC, PSNR, SSIM: higher is better
        # Allow 10% margin of error
        
        margin = 0.90  # Agent should achieve at least 90% of standard
        
        cc_ok = agent_cc >= std_cc * margin or agent_cc >= std_cc - 0.05
        psnr_ok = agent_psnr >= std_psnr * margin or agent_psnr >= std_psnr - 2.0
        ssim_ok = agent_ssim >= std_ssim * margin or agent_ssim >= std_ssim - 0.05
        
        all_ok = cc_ok and psnr_ok and ssim_ok
        
        if all_ok:
            print("\n[PASS] Agent's performance is acceptable.")
            print(f"  CC: {'PASS' if cc_ok else 'FAIL'}")
            print(f"  PSNR: {'PASS' if psnr_ok else 'FAIL'}")
            print(f"  SSIM: {'PASS' if ssim_ok else 'FAIL'}")
            sys.exit(0)
        else:
            print("\n[FAIL] Agent's performance degraded significantly.")
            print(f"  CC: {'PASS' if cc_ok else 'FAIL'} (Agent: {agent_cc:.4f}, Std: {std_cc:.4f})")
            print(f"  PSNR: {'PASS' if psnr_ok else 'FAIL'} (Agent: {agent_psnr:.2f}, Std: {std_psnr:.2f})")
            print(f"  SSIM: {'PASS' if ssim_ok else 'FAIL'} (Agent: {agent_ssim:.4f}, Std: {std_ssim:.4f})")
            sys.exit(1)
    
    except Exception as e:
        print(f"[ERROR] Exception occurred during testing:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()