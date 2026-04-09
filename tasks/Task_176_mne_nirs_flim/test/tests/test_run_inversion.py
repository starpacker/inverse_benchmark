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

# Setup directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Inject the referee (evaluate_results) verbatim from Reference B
def evaluate_results(hbo_gt, hbr_gt, hbo_rec, hbr_rec, t, od_760_noisy, od_850_noisy,
                     block_starts, block_duration, save_plots=True, save_metrics=True):
    """
    Compute metrics for both HbO and HbR, create visualizations, and save results.
    
    Parameters
    ----------
    hbo_gt : ndarray
        Ground truth HbO concentration
    hbr_gt : ndarray
        Ground truth HbR concentration
    hbo_rec : ndarray
        Recovered HbO concentration
    hbr_rec : ndarray
        Recovered HbR concentration
    t : ndarray
        Time vector
    od_760_noisy : ndarray
        Noisy optical density at 760nm
    od_850_noisy : ndarray
        Noisy optical density at 850nm
    block_starts : list
        Start times of stimulus blocks
    block_duration : float
        Duration of each stimulus block
    save_plots : bool
        Whether to save plots
    save_metrics : bool
        Whether to save metrics to JSON
        
    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics
    """
    # Compute PSNR
    def compute_psnr(gt, rec):
        mse = np.mean((gt - rec) ** 2)
        if mse < 1e-30:
            return 100.0
        peak = np.max(np.abs(gt))
        return 10 * np.log10(peak ** 2 / mse)
    
    # Compute correlation coefficient
    def compute_cc(gt, rec):
        return float(np.corrcoef(gt, rec)[0, 1])
    
    # Compute RMSE
    def compute_rmse(gt, rec):
        return float(np.sqrt(np.mean((gt - rec) ** 2)))
    
    metrics = {
        'HbO_PSNR_dB': compute_psnr(hbo_gt, hbo_rec),
        'HbO_CC': compute_cc(hbo_gt, hbo_rec),
        'HbO_RMSE': compute_rmse(hbo_gt, hbo_rec),
        'HbR_PSNR_dB': compute_psnr(hbr_gt, hbr_rec),
        'HbR_CC': compute_cc(hbr_gt, hbr_rec),
        'HbR_RMSE': compute_rmse(hbr_gt, hbr_rec),
    }
    
    # Overall averages
    metrics['PSNR_dB'] = (metrics['HbO_PSNR_dB'] + metrics['HbR_PSNR_dB']) / 2
    metrics['CC'] = (metrics['HbO_CC'] + metrics['HbR_CC']) / 2
    metrics['RMSE'] = (metrics['HbO_RMSE'] + metrics['HbR_RMSE']) / 2
    
    # Print metrics
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    
    # Save metrics
    if save_metrics:
        metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Metrics saved to {metrics_path}")
    
    # Create visualization
    if save_plots:
        fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
        
        # Helper: shade stimulus blocks
        def shade_blocks(ax):
            for s in block_starts:
                ax.axvspan(s, s + block_duration, color='yellow', alpha=0.2, label=None)
        
        # Panel 1: Ground truth HbO & HbR
        ax = axes[0]
        shade_blocks(ax)
        ax.plot(t, hbo_gt * 1e6, 'r-', lw=1.5, label='HbO (GT)')
        ax.plot(t, hbr_gt * 1e6, 'b-', lw=1.5, label='HbR (GT)')
        ax.set_ylabel('Concentration (µM)')
        ax.set_title('Ground Truth Hemodynamic Response')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Noisy optical density input
        ax = axes[1]
        shade_blocks(ax)
        ax.plot(t, od_760_noisy, 'purple', lw=0.8, alpha=0.7, label='ΔOD 760nm')
        ax.plot(t, od_850_noisy, 'orange', lw=0.8, alpha=0.7, label='ΔOD 850nm')
        ax.set_ylabel('Optical Density Change')
        ax.set_title('Noisy Optical Density Input (MBLL Forward + Noise)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Recovered vs GT
        ax = axes[2]
        shade_blocks(ax)
        ax.plot(t, hbo_gt * 1e6, 'r--', lw=1.5, alpha=0.6, label='HbO (GT)')
        ax.plot(t, hbo_rec * 1e6, 'r-', lw=1.2, label='HbO (Recovered)')
        ax.plot(t, hbr_gt * 1e6, 'b--', lw=1.5, alpha=0.6, label='HbR (GT)')
        ax.plot(t, hbr_rec * 1e6, 'b-', lw=1.2, label='HbR (Recovered)')
        ax.set_ylabel('Concentration (µM)')
        ax.set_title(f'Recovered Hemodynamic Response  |  PSNR={metrics["PSNR_dB"]:.1f} dB, CC={metrics["CC"]:.4f}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Panel 4: Residuals
        ax = axes[3]
        shade_blocks(ax)
        ax.plot(t, (hbo_gt - hbo_rec) * 1e6, 'r-', lw=1.0, label='HbO residual')
        ax.plot(t, (hbr_gt - hbr_rec) * 1e6, 'b-', lw=1.0, label='HbR residual')
        ax.axhline(0, color='k', ls='--', lw=0.5)
        ax.set_ylabel('Residual (µM)')
        ax.set_xlabel('Time (s)')
        ax.set_title('Residuals (GT − Recovered)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = os.path.join(RESULTS_DIR, 'reconstruction_result.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Figure saved to {fig_path}")
    
    # Save arrays
    gt_stack = np.stack([hbo_gt, hbr_gt], axis=0)
    rec_stack = np.stack([hbo_rec, hbr_rec], axis=0)
    input_stack = np.stack([od_760_noisy, od_850_noisy], axis=0)
    
    np.save(os.path.join(RESULTS_DIR, 'ground_truth.npy'), gt_stack)
    np.save(os.path.join(RESULTS_DIR, 'reconstruction.npy'), rec_stack)
    np.save(os.path.join(RESULTS_DIR, 'input_data.npy'), input_stack)
    print(f"  Arrays saved to {RESULTS_DIR}")
    
    return metrics


def main():
    # Data paths provided
    data_paths = ['/data/yjh/mne_nirs_flim_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer and inner data files
    outer_data_files = []
    inner_data_files = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_data_files.append(path)
        else:
            outer_data_files.append(path)
    
    print(f"Outer data files: {outer_data_files}")
    print(f"Inner data files: {inner_data_files}")
    
    try:
        # Load the primary (outer) data
        if not outer_data_files:
            print("ERROR: No outer data file found!")
            sys.exit(1)
        
        outer_path = outer_data_files[0]
        print(f"\nLoading outer data from: {outer_path}")
        
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"Outer data keys: {outer_data.keys()}")
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"Running run_inversion with {len(args)} args and {len(kwargs)} kwargs")
        
        # Execute the agent function
        agent_output = run_inversion(*args, **kwargs)
        
        # Check if this is a chained execution (function returns a callable)
        if inner_data_files and callable(agent_output):
            # Chained execution pattern
            inner_path = inner_data_files[0]
            print(f"\nLoading inner data from: {inner_path}")
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            print(f"Running inner function with {len(inner_args)} args and {len(inner_kwargs)} kwargs")
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Direct execution pattern
            final_result = agent_output
            std_result = std_output
        
        print("\n--- Agent Output ---")
        print(f"Type: {type(final_result)}")
        if isinstance(final_result, dict):
            print(f"Keys: {final_result.keys()}")
        
        print("\n--- Standard Output ---")
        print(f"Type: {type(std_result)}")
        if isinstance(std_result, dict):
            print(f"Keys: {std_result.keys()}")
        
        # Extract hbo_rec and hbr_rec from both results
        if isinstance(final_result, dict):
            agent_hbo_rec = final_result.get('hbo_rec')
            agent_hbr_rec = final_result.get('hbr_rec')
        else:
            print("ERROR: Agent output is not a dict!")
            sys.exit(1)
        
        if isinstance(std_result, dict):
            std_hbo_rec = std_result.get('hbo_rec')
            std_hbr_rec = std_result.get('hbr_rec')
        else:
            print("ERROR: Standard output is not a dict!")
            sys.exit(1)
        
        # For evaluation, we need ground truth data
        # The evaluate_results function requires hbo_gt, hbr_gt, t, od_760_noisy, od_850_noisy, etc.
        # These should be in the input args
        
        # Extract input data from args
        od_760_noisy = args[0] if len(args) > 0 else kwargs.get('od_760_noisy')
        od_850_noisy = args[1] if len(args) > 1 else kwargs.get('od_850_noisy')
        
        # Generate synthetic ground truth and time parameters for evaluation
        # Since we don't have the actual GT, we'll compare agent vs standard directly
        # by using the standard result as a proxy for GT
        
        # Create time vector
        n_samples = len(od_760_noisy) if od_760_noisy is not None else 1000
        t = np.linspace(0, 100, n_samples)  # Assume 100 seconds
        
        # Use default block parameters
        block_starts = [10, 40, 70]  # Default stimulus blocks
        block_duration = 15.0
        
        print("\n=== Evaluating Agent Results ===")
        # Use standard result as ground truth for comparison
        agent_metrics = evaluate_results(
            hbo_gt=std_hbo_rec,
            hbr_gt=std_hbr_rec,
            hbo_rec=agent_hbo_rec,
            hbr_rec=agent_hbr_rec,
            t=t,
            od_760_noisy=od_760_noisy,
            od_850_noisy=od_850_noisy,
            block_starts=block_starts,
            block_duration=block_duration,
            save_plots=True,
            save_metrics=True
        )
        
        # Extract key metrics
        agent_psnr = agent_metrics['PSNR_dB']
        agent_cc = agent_metrics['CC']
        agent_rmse = agent_metrics['RMSE']
        
        print(f"\n=== Final Scores ===")
        print(f"Agent PSNR: {agent_psnr:.4f} dB")
        print(f"Agent CC: {agent_cc:.6f}")
        print(f"Agent RMSE: {agent_rmse:.10f}")
        
        # Verify performance
        # Since we compare agent to standard, perfect match should give very high PSNR
        # A PSNR > 60 dB indicates nearly identical results
        # CC should be very close to 1.0
        
        PSNR_THRESHOLD = 50.0  # dB - high threshold since we expect near-identical results
        CC_THRESHOLD = 0.999
        
        success = True
        
        if agent_psnr < PSNR_THRESHOLD:
            print(f"WARNING: PSNR {agent_psnr:.2f} dB below threshold {PSNR_THRESHOLD} dB")
            # For near-identical comparisons, this might be acceptable
            if agent_psnr < 30.0:
                print("FAIL: PSNR critically low - significant deviation from standard")
                success = False
        
        if agent_cc < CC_THRESHOLD:
            print(f"WARNING: CC {agent_cc:.6f} below threshold {CC_THRESHOLD}")
            if agent_cc < 0.99:
                print("FAIL: Correlation critically low")
                success = False
        
        # Additional direct comparison
        hbo_diff = np.max(np.abs(agent_hbo_rec - std_hbo_rec))
        hbr_diff = np.max(np.abs(agent_hbr_rec - std_hbr_rec))
        
        print(f"\nDirect comparison:")
        print(f"  Max HbO difference: {hbo_diff:.2e}")
        print(f"  Max HbR difference: {hbr_diff:.2e}")
        
        # For numerical algorithms, very small differences are acceptable
        if hbo_diff < 1e-10 and hbr_diff < 1e-10:
            print("Results are numerically identical (diff < 1e-10)")
            success = True
        
        if success:
            print("\n=== TEST PASSED ===")
            sys.exit(0)
        else:
            print("\n=== TEST FAILED ===")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()