import sys
import os
import dill
import numpy as np
import traceback

# Import target function
from agent_run_inversion import run_inversion

# Import dependencies required by evaluate_results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from skimage.metrics import structural_similarity as ssim_fn

# Setup directories
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Inject the referee function (evaluate_results) verbatim from Reference B
def evaluate_results(data, result, save_dir):
    """
    Compute metrics, save results, and generate visualizations.
    
    Parameters:
        data: Dictionary containing ground truth data
        result: Dictionary containing reconstruction results
        save_dir: Directory to save results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    sig_gt = data['sigma_xx_gt']
    sig_rec = result['sigma_xx_rec']
    E_gt = data['gt_E']
    E_rec = result['E_rec']
    nu_gt = data['gt_nu']
    nu_rec = result['nu_rec']
    xx = data['xx']
    yy = data['yy']
    
    # Compute metrics
    dr = sig_gt.max() - sig_gt.min()
    if dr < 1e-12:
        dr = 1.0
    mse = np.mean((sig_gt - sig_rec)**2)
    psnr = float(10*np.log10(dr**2/max(mse, 1e-30)))
    ssim_val = float(ssim_fn(sig_gt, sig_rec, data_range=dr))
    cc = float(np.corrcoef(sig_gt.ravel(), sig_rec.ravel())[0, 1])
    re = float(np.linalg.norm(sig_gt - sig_rec)/max(np.linalg.norm(sig_gt), 1e-12))
    
    metrics = {
        "PSNR": psnr,
        "SSIM": ssim_val,
        "CC": cc,
        "RE": re,
        "E_gt": float(E_gt),
        "E_rec": float(E_rec),
        "E_err_pct": float(abs(E_gt - E_rec)/E_gt*100),
        "nu_gt": float(nu_gt),
        "nu_rec": float(nu_rec),
        "nu_err_pct": float(abs(nu_gt - nu_rec)/nu_gt*100)
    }
    
    # Print metrics
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")
    
    # Save metrics to JSON
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save reconstructions
    np.save(os.path.join(save_dir, "reconstruction.npy"), sig_rec)
    np.save(os.path.join(save_dir, "ground_truth.npy"), sig_gt)
    
    # Generate visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmax = max(np.abs(sig_gt).max(), np.abs(sig_rec).max())
    
    for ax, arr, title in zip(axes[:2], [sig_gt, sig_rec],
                               ['GT σ_xx', 'VFM σ_xx']):
        im = ax.imshow(arr.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       origin='lower', aspect='auto')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    
    err = sig_gt - sig_rec
    im = axes[2].imshow(err.T, cmap='RdBu_r', origin='lower', aspect='auto')
    axes[2].set_title('Error')
    plt.colorbar(im, ax=axes[2])
    
    fig.suptitle(f"VFM — E={metrics['E_rec']:.0f} MPa (err {metrics['E_err_pct']:.1f}%)  |  "
                 f"ν={metrics['nu_rec']:.3f} (err {metrics['nu_err_pct']:.1f}%)  |  "
                 f"CC={metrics['CC']:.4f}", fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    save_path = os.path.join(save_dir, "reconstruction_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")
    
    return metrics


def main():
    # Data paths provided
    data_paths = ['/data/yjh/VFM_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_files = []
    inner_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_files.append(path)
        else:
            outer_files.append(path)
    
    print(f"Outer files: {outer_files}")
    print(f"Inner files: {inner_files}")
    
    # Determine execution pattern
    is_chained = len(inner_files) > 0
    
    try:
        # Load primary (outer) data
        if not outer_files:
            print("ERROR: No outer data file found!")
            sys.exit(1)
        
        outer_path = outer_files[0]
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"Outer data keys: {outer_data.keys()}")
        
        # Extract args and kwargs
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Running run_inversion with args of length {len(args)}")
        
        # Execute the agent function
        agent_output = run_inversion(*args, **kwargs)
        
        if is_chained:
            # Pattern 2: Chained Execution
            inner_path = inner_files[0]
            print(f"Loading inner data from: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # agent_output should be callable
            if callable(agent_output):
                final_result = agent_output(*inner_args, **inner_kwargs)
            else:
                print("WARNING: agent_output is not callable in chained mode. Using as-is.")
                final_result = agent_output
        else:
            # Pattern 1: Direct Execution
            final_result = agent_output
            std_result = std_output
        
        # The input data for evaluation is the first argument (data dictionary)
        input_data = args[0] if args else kwargs.get('data', {})
        
        print("\n" + "="*60)
        print("EVALUATING AGENT RESULTS")
        print("="*60)
        
        # Create separate save directories for agent and standard
        agent_save_dir = os.path.join(RESULTS_DIR, "agent_results")
        std_save_dir = os.path.join(RESULTS_DIR, "standard_results")
        os.makedirs(agent_save_dir, exist_ok=True)
        os.makedirs(std_save_dir, exist_ok=True)
        
        # Evaluate agent results
        metrics_agent = evaluate_results(input_data, final_result, agent_save_dir)
        
        print("\n" + "="*60)
        print("EVALUATING STANDARD RESULTS")
        print("="*60)
        
        # Evaluate standard results
        metrics_std = evaluate_results(input_data, std_result, std_save_dir)
        
        # Extract primary metrics for comparison
        # Using CC (Correlation Coefficient) as primary metric - higher is better
        score_agent = metrics_agent['CC']
        score_std = metrics_std['CC']
        
        # Also check PSNR and RE for additional validation
        psnr_agent = metrics_agent['PSNR']
        psnr_std = metrics_std['PSNR']
        re_agent = metrics_agent['RE']
        re_std = metrics_std['RE']
        e_err_agent = metrics_agent['E_err_pct']
        e_err_std = metrics_std['E_err_pct']
        nu_err_agent = metrics_agent['nu_err_pct']
        nu_err_std = metrics_std['nu_err_pct']
        
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"Scores -> Agent CC: {score_agent:.6f}, Standard CC: {score_std:.6f}")
        print(f"Scores -> Agent PSNR: {psnr_agent:.4f}, Standard PSNR: {psnr_std:.4f}")
        print(f"Scores -> Agent RE: {re_agent:.6f}, Standard RE: {re_std:.6f}")
        print(f"Scores -> Agent E_err%: {e_err_agent:.4f}, Standard E_err%: {e_err_std:.4f}")
        print(f"Scores -> Agent nu_err%: {nu_err_agent:.4f}, Standard nu_err%: {nu_err_std:.4f}")
        
        # Determine success
        # CC: Higher is better - agent should be at least 90% of standard
        # PSNR: Higher is better - agent should be at least 90% of standard
        # RE: Lower is better - agent should be at most 110% of standard
        
        margin = 0.10  # 10% margin
        
        cc_ok = score_agent >= score_std * (1 - margin) or score_agent >= 0.99
        psnr_ok = psnr_agent >= psnr_std * (1 - margin) or psnr_agent >= 30
        re_ok = re_agent <= re_std * (1 + margin) or re_agent <= 0.05
        
        print(f"\nValidation Results:")
        print(f"  CC check: {'PASS' if cc_ok else 'FAIL'} (Agent: {score_agent:.6f}, Threshold: {score_std * (1 - margin):.6f})")
        print(f"  PSNR check: {'PASS' if psnr_ok else 'FAIL'} (Agent: {psnr_agent:.4f}, Threshold: {psnr_std * (1 - margin):.4f})")
        print(f"  RE check: {'PASS' if re_ok else 'FAIL'} (Agent: {re_agent:.6f}, Threshold: {re_std * (1 + margin):.6f})")
        
        # Overall pass if at least 2 out of 3 metrics pass, or CC is very high
        passes = sum([cc_ok, psnr_ok, re_ok])
        
        if passes >= 2 or (score_agent >= 0.999):
            print("\n✓ TEST PASSED: Agent performance is acceptable.")
            sys.exit(0)
        else:
            print("\n✗ TEST FAILED: Agent performance degraded significantly.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nERROR during testing: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()