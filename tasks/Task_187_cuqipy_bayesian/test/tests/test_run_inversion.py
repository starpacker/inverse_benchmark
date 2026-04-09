import sys
import os
import dill
import numpy as np
import traceback
import json
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Import target function
from agent_run_inversion import run_inversion

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============== INJECT REFEREE (evaluate_results) ==============
def forward_operator(x, forward_model):
    """
    Apply the forward convolution operator to a signal.
    
    Parameters
    ----------
    x : numpy.ndarray
        Input signal of shape (dim,).
    forward_model : cuqi.model.LinearModel
        Linear forward convolution model from CUQIpy.
    
    Returns
    -------
    numpy.ndarray
        Convolved signal (predicted observations).
    """
    y_pred = forward_model @ x
    return np.asarray(y_pred)

def evaluate_results(data_dict, inversion_result, results_dir):
    """
    Evaluate reconstruction quality and save results.
    
    Computes PSNR and SSIM for all reconstructions, selects the best one,
    saves metrics and visualizations.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary from load_and_preprocess_data.
    inversion_result : dict
        Dictionary from run_inversion.
    results_dir : str
        Directory to save results.
    
    Returns
    -------
    tuple
        (best_psnr, best_ssim) - metrics of the best reconstruction.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    x_true = data_dict['x_true']
    y_data = data_dict['y_data']
    A = data_dict['forward_model']
    dim = data_dict['dim']
    phantom = data_dict['phantom']
    noise_std = data_dict['noise_std']
    
    x_map_gmrf = inversion_result['x_map_gmrf']
    x_map_lmrf = inversion_result['x_map_lmrf']
    posterior_mean = inversion_result['posterior_mean']
    lower_ci = inversion_result['lower_ci']
    upper_ci = inversion_result['upper_ci']
    samples_array = inversion_result['samples']
    n_samples = inversion_result['n_samples']
    gmrf_precision = inversion_result['gmrf_precision']
    lmrf_precision = inversion_result['lmrf_precision']
    
    # Helper function for metrics
    def compute_metrics(ground_truth, reconstruction):
        gt = np.asarray(ground_truth, dtype=np.float64)
        rec = np.asarray(reconstruction, dtype=np.float64)
        data_range = gt.max() - gt.min()
        psnr = peak_signal_noise_ratio(gt, rec, data_range=data_range)
        ssim = structural_similarity(gt, rec, data_range=data_range)
        return psnr, ssim
    
    # Compute metrics for GMRF MAP
    psnr_map, ssim_map = compute_metrics(x_true, x_map_gmrf)
    print(f"  GMRF MAP PSNR: {psnr_map:.2f} dB")
    print(f"  GMRF MAP SSIM: {ssim_map:.4f}")
    
    # Compute metrics for posterior mean
    psnr_mean, ssim_mean = compute_metrics(x_true, posterior_mean)
    print(f"  Posterior mean PSNR: {psnr_mean:.2f} dB")
    print(f"  Posterior mean SSIM: {ssim_mean:.4f}")
    
    # Compute metrics for LMRF MAP if available
    psnr_lmrf, ssim_lmrf = 0.0, 0.0
    if x_map_lmrf is not None:
        psnr_lmrf, ssim_lmrf = compute_metrics(x_true, x_map_lmrf)
        print(f"  LMRF MAP PSNR: {psnr_lmrf:.2f} dB, SSIM: {ssim_lmrf:.4f}")
    
    # Select best reconstruction
    if psnr_mean >= psnr_map:
        x_recon = posterior_mean
        psnr_val, ssim_val = psnr_mean, ssim_mean
        recon_label = "Posterior Mean"
    else:
        x_recon = x_map_gmrf
        psnr_val, ssim_val = psnr_map, ssim_map
        recon_label = "MAP"
    
    # Check if LMRF is better
    if x_map_lmrf is not None and psnr_lmrf > psnr_val:
        x_recon = x_map_lmrf
        psnr_val, ssim_val = psnr_lmrf, ssim_lmrf
        recon_label = "LMRF MAP"
        print(f"  -> LMRF MAP is better, using it.")
    
    print(f"\nBest reconstruction: {recon_label}")
    print(f"  PSNR: {psnr_val:.2f} dB")
    print(f"  SSIM: {ssim_val:.4f}")
    
    # Save metrics
    metrics = {
        "psnr": round(float(psnr_val), 4),
        "ssim": round(float(ssim_val), 4),
        "psnr_map_gmrf": round(float(psnr_map), 4),
        "ssim_map_gmrf": round(float(ssim_map), 4),
        "psnr_posterior_mean": round(float(psnr_mean), 4),
        "ssim_posterior_mean": round(float(ssim_mean), 4),
        "psnr_map_lmrf": round(float(psnr_lmrf), 4),
        "ssim_map_lmrf": round(float(ssim_lmrf), 4),
        "best_method": recon_label,
        "n_posterior_samples": n_samples,
        "dim": dim,
        "phantom": phantom,
        "noise_std": noise_std,
        "gmrf_precision": gmrf_precision,
        "lmrf_precision": lmrf_precision,
    }
    
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    np.save(os.path.join(results_dir, "ground_truth.npy"), x_true)
    np.save(os.path.join(results_dir, "reconstruction.npy"), x_recon)
    
    print(f"\nMetrics saved to {results_dir}/metrics.json")
    
    # Visualization
    grid = np.linspace(0, 1, dim)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (a) Ground Truth Signal
    ax = axes[0, 0]
    ax.plot(grid, x_true, 'k-', linewidth=2, label='Ground Truth')
    ax.set_title('(a) Ground Truth Signal', fontsize=13, fontweight='bold')
    ax.set_xlabel('Position')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (b) Noisy Observations
    ax = axes[0, 1]
    ax.plot(grid, y_data, 'r-', linewidth=1, alpha=0.8, label='Noisy Observations')
    ax.plot(grid, forward_operator(x_true, A), 'b--', linewidth=1.5, alpha=0.6, label='Clean Convolved')
    ax.set_title('(b) Observed Data (Convolved + Noise)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Position')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (c) MAP Reconstruction
    ax = axes[1, 0]
    ax.plot(grid, x_true, 'k-', linewidth=2, alpha=0.5, label='Ground Truth')
    ax.plot(grid, x_recon, 'b-', linewidth=2, label=f'{recon_label} (Best)')
    if recon_label != "MAP":
        ax.plot(grid, x_map_gmrf, 'g--', linewidth=1.5, alpha=0.6, label='GMRF MAP')
    ax.set_title(f'(c) {recon_label} Reconstruction\nPSNR={psnr_val:.2f} dB, SSIM={ssim_val:.4f}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Position')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (d) Posterior Mean + 95% Credible Interval
    ax = axes[1, 1]
    ax.plot(grid, x_true, 'k-', linewidth=2, alpha=0.5, label='Ground Truth')
    ax.plot(grid, posterior_mean, 'b-', linewidth=2, label='Posterior Mean')
    ax.fill_between(grid, lower_ci, upper_ci, alpha=0.25, color='blue',
                    label='95% Credible Interval')
    # Plot a few individual samples
    for i in range(min(10, n_samples)):
        ax.plot(grid, samples_array[:, i], color='steelblue', alpha=0.08, linewidth=0.5)
    ax.set_title(f'(d) Posterior Mean + 95% CI ({n_samples} samples)\n'
                 f'PSNR={psnr_mean:.2f} dB, SSIM={ssim_mean:.4f}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Position')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('CUQIpy: Bayesian 1D Deconvolution with Uncertainty Quantification',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "reconstruction_result.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {results_dir}/reconstruction_result.png")
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS: PSNR = {psnr_val:.4f} dB, SSIM = {ssim_val:.4f}")
    print(f"{'='*50}")
    
    return psnr_val, ssim_val


# ============== MAIN TEST LOGIC ==============
def main():
    data_paths = ['/data/yjh/cuqipy_bayesian_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
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
    
    # Load outer (primary) data
    if not outer_data_files:
        print("ERROR: No outer data file found.")
        sys.exit(1)
    
    outer_data_path = outer_data_files[0]
    print(f"\nLoading outer data from: {outer_data_path}")
    
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Outer function name: {outer_data.get('func_name', 'unknown')}")
    print(f"Number of args: {len(args)}")
    print(f"Kwargs keys: {list(kwargs.keys())}")
    
    # Run the agent's function
    print("\n--- Running agent's run_inversion ---")
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"ERROR running agent's run_inversion: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if chained execution (inner data exists)
    if inner_data_files:
        # Chained execution pattern
        inner_data_path = inner_data_files[0]
        print(f"\nLoading inner data from: {inner_data_path}")
        
        try:
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"ERROR loading inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        print(f"Inner function name: {inner_data.get('func_name', 'unknown')}")
        
        # agent_output should be callable
        if callable(agent_output):
            print("\n--- Running chained function (agent_output as operator) ---")
            try:
                final_agent_result = agent_output(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR running chained function: {e}")
                traceback.print_exc()
                sys.exit(1)
        else:
            print("ERROR: Expected agent_output to be callable for chained execution.")
            sys.exit(1)
    else:
        # Direct execution pattern
        final_agent_result = agent_output
        std_result = std_output
    
    # Extract data_dict from args for evaluation
    # data_dict is the first argument to run_inversion
    if len(args) > 0:
        data_dict = args[0]
    elif 'data_dict' in kwargs:
        data_dict = kwargs['data_dict']
    else:
        print("ERROR: Cannot find data_dict in args or kwargs.")
        sys.exit(1)
    
    # Create results directories
    results_dir_agent = './results_agent'
    results_dir_std = './results_std'
    
    print("\n" + "="*60)
    print("EVALUATING AGENT'S RESULT")
    print("="*60)
    
    try:
        score_agent = evaluate_results(data_dict, final_agent_result, results_dir_agent)
        psnr_agent, ssim_agent = score_agent
    except Exception as e:
        print(f"ERROR evaluating agent result: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*60)
    print("EVALUATING STANDARD RESULT")
    print("="*60)
    
    try:
        score_std = evaluate_results(data_dict, std_result, results_dir_std)
        psnr_std, ssim_std = score_std
    except Exception as e:
        print(f"ERROR evaluating standard result: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Final comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Agent PSNR: {psnr_agent:.4f} dB, SSIM: {ssim_agent:.4f}")
    print(f"Standard PSNR: {psnr_std:.4f} dB, SSIM: {ssim_std:.4f}")
    
    # PSNR and SSIM: Higher is better
    # Allow 10% margin of error
    margin = 0.90
    
    psnr_pass = psnr_agent >= psnr_std * margin
    ssim_pass = ssim_agent >= ssim_std * margin
    
    print(f"\nPSNR check (agent >= {margin}*std): {psnr_agent:.4f} >= {psnr_std * margin:.4f} -> {'PASS' if psnr_pass else 'FAIL'}")
    print(f"SSIM check (agent >= {margin}*std): {ssim_agent:.4f} >= {ssim_std * margin:.4f} -> {'PASS' if ssim_pass else 'FAIL'}")
    
    if psnr_pass and ssim_pass:
        print("\n*** TEST PASSED: Agent performance is acceptable. ***")
        sys.exit(0)
    else:
        print("\n*** TEST FAILED: Agent performance degraded significantly. ***")
        sys.exit(1)


if __name__ == "__main__":
    main()