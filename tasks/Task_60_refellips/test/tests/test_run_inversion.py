import sys
import os
import dill
import numpy as np
import traceback

# Import the target function from the agent module
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from skimage.metrics import structural_similarity as ssim_fn

# Inject the referee function (evaluate_results) verbatim from Reference B
def cauchy_n(wavelength_nm, A, B, C):
    """Cauchy dispersion: n(λ) = A + B/λ² + C/λ⁴"""
    lam_um = wavelength_nm / 1000.0
    return A + B / lam_um**2 + C / lam_um**4

def evaluate_results(data, inversion_result, results_dir):
    """
    Compute metrics, save outputs, and visualize results.

    Parameters
    ----------
    data : dict
        Dictionary containing wavelengths, clean data, gt_params, etc.
    inversion_result : dict
        Dictionary containing fit_params, psi_fit, delta_fit.
    results_dir : str
        Directory to save results.

    Returns
    -------
    metrics : dict
        Dictionary of computed metrics.
    """
    wavelengths = data["wavelengths"]
    psi_clean = data["psi_clean"]
    delta_clean = data["delta_clean"]
    psi_meas = data["psi_noisy"]
    delta_meas = data["delta_noisy"]
    gt = data["gt_params"]

    fit = inversion_result["fit_params"]
    psi_fit = inversion_result["psi_fit"]
    delta_fit = inversion_result["delta_fit"]

    # Ψ metrics
    rmse_psi = float(np.sqrt(np.mean((psi_clean - psi_fit)**2)))
    cc_psi = float(np.corrcoef(psi_clean, psi_fit)[0, 1])

    # Δ metrics
    rmse_delta = float(np.sqrt(np.mean((delta_clean - delta_fit)**2)))
    cc_delta = float(np.corrcoef(delta_clean, delta_fit)[0, 1])

    # Combined PSNR/SSIM on Ψ
    dr = psi_clean.max() - psi_clean.min()
    mse = np.mean((psi_clean - psi_fit)**2)
    psnr = float(10 * np.log10(dr**2 / max(mse, 1e-30)))
    tile_rows = 7
    a2d = np.tile(psi_clean, (tile_rows, 1))
    b2d = np.tile(psi_fit, (tile_rows, 1))
    ssim_val = float(ssim_fn(a2d, b2d, data_range=dr, win_size=7))

    # Parameter recovery
    param_metrics = {}
    for k in ["thickness", "A", "B", "C", "k_amp"]:
        g, f = gt[k], fit[k]
        param_metrics[f"gt_{k}"] = float(g)
        param_metrics[f"fit_{k}"] = float(f)
        param_metrics[f"abs_err_{k}"] = float(abs(g - f))

    # n(λ) recovery
    n_gt = cauchy_n(wavelengths, gt["A"], gt["B"], gt["C"])
    n_fit = cauchy_n(wavelengths, fit["A"], fit["B"], fit["C"])
    cc_n = float(np.corrcoef(n_gt, n_fit)[0, 1])

    metrics = {
        "PSNR_psi": psnr,
        "SSIM_psi": ssim_val,
        "CC_psi": cc_psi,
        "RMSE_psi_deg": rmse_psi,
        "CC_delta": cc_delta,
        "RMSE_delta_deg": rmse_delta,
        "CC_n": cc_n,
        **param_metrics,
    }

    # Print metrics
    for k, v in sorted(metrics.items()):
        print(f"  {k:30s} = {v}")

    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save reconstructions
    np.save(
        os.path.join(results_dir, "reconstruction.npy"),
        np.column_stack([psi_fit, delta_fit])
    )
    np.save(
        os.path.join(results_dir, "ground_truth.npy"),
        np.column_stack([psi_clean, delta_clean])
    )

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(wavelengths, psi_clean, 'b-', lw=2, label='GT')
    ax.plot(wavelengths, psi_meas, 'k.', ms=2, alpha=0.3, label='Noisy')
    ax.plot(wavelengths, psi_fit, 'r--', lw=1.5, label='Fit')
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('Ψ [°]')
    ax.set_title('(a) Ψ(λ)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(wavelengths, delta_clean, 'b-', lw=2, label='GT')
    ax.plot(wavelengths, delta_meas, 'k.', ms=2, alpha=0.3, label='Noisy')
    ax.plot(wavelengths, delta_fit, 'r--', lw=1.5, label='Fit')
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('Δ [°]')
    ax.set_title('(b) Δ(λ)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    n_gt_plot = cauchy_n(wavelengths, gt["A"], gt["B"], gt["C"])
    n_fit_plot = cauchy_n(wavelengths, fit["A"], fit["B"], fit["C"])
    
    ax = axes[1, 0]
    ax.plot(wavelengths, n_gt_plot, 'b-', lw=2, label='GT n(λ)')
    ax.plot(wavelengths, n_fit_plot, 'r--', lw=2, label='Fit n(λ)')
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('Refractive index n')
    ax.set_title('(c) Dispersion')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    labels = ['d [nm]', 'A', 'B', 'C', 'k_amp']
    keys = ['thickness', 'A', 'B', 'C', 'k_amp']
    gt_v = [gt[k] for k in keys]
    fit_v = [fit[k] for k in keys]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, gt_v, w, label='GT', color='steelblue')
    ax.bar(x + w/2, fit_v, w, label='Fit', color='tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title('(d) Parameters')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        f"refellips — Spectroscopic Ellipsometry Inversion\n"
        f"PSNR(Ψ)={metrics['PSNR_psi']:.1f} dB  |  CC(Ψ)={metrics['CC_psi']:.4f}  |  "
        f"CC(Δ)={metrics['CC_delta']:.4f}  |  Δd={metrics['abs_err_thickness']:.2f} nm",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    save_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")

    return metrics


def main():
    # Data paths
    data_paths = ['/data/yjh/refellips_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Results directory for evaluation outputs
    results_dir = './test_results'
    os.makedirs(results_dir, exist_ok=True)
    
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
    
    try:
        # Load the primary (outer) data
        if not outer_files:
            print("[ERROR] No outer data file found!")
            sys.exit(1)
        
        outer_path = outer_files[0]
        print(f"[INFO] Loading outer data from: {outer_path}")
        
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"[INFO] Outer data keys: {outer_data.keys()}")
        
        # Extract args and kwargs from outer data
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"[INFO] Running agent's run_inversion with args and kwargs...")
        
        # Run the agent's function
        agent_output = run_inversion(*args, **kwargs)
        
        # Check if we have inner data (chained execution pattern)
        if inner_files:
            print(f"[INFO] Chained execution detected. Processing inner data...")
            inner_path = inner_files[0]
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            # Execute the returned operator with inner data
            if callable(agent_output):
                final_result = agent_output(*inner_args, **inner_kwargs)
            else:
                print("[WARNING] Agent output is not callable. Using direct output.")
                final_result = agent_output
        else:
            # Direct execution pattern
            final_result = agent_output
            std_result = std_output
        
        print("[INFO] Agent execution completed.")
        print(f"[INFO] Agent result keys: {final_result.keys() if isinstance(final_result, dict) else type(final_result)}")
        print(f"[INFO] Standard result keys: {std_result.keys() if isinstance(std_result, dict) else type(std_result)}")
        
        # Extract the input data for evaluation
        # The 'data' dict should be the first argument (args[0])
        input_data = args[0] if args else kwargs.get('data', None)
        
        if input_data is None:
            print("[ERROR] Could not extract input data for evaluation!")
            sys.exit(1)
        
        # Evaluate agent's result
        print("\n[INFO] Evaluating Agent's result...")
        agent_results_dir = os.path.join(results_dir, 'agent')
        metrics_agent = evaluate_results(input_data, final_result, agent_results_dir)
        
        # Evaluate standard result
        print("\n[INFO] Evaluating Standard result...")
        std_results_dir = os.path.join(results_dir, 'standard')
        metrics_std = evaluate_results(input_data, std_result, std_results_dir)
        
        # Extract primary metrics for comparison
        # Using PSNR_psi as the primary metric (higher is better)
        score_agent = metrics_agent.get('PSNR_psi', 0.0)
        score_std = metrics_std.get('PSNR_psi', 0.0)
        
        # Also compare CC (correlation coefficient) metrics
        cc_psi_agent = metrics_agent.get('CC_psi', 0.0)
        cc_psi_std = metrics_std.get('CC_psi', 0.0)
        
        cc_delta_agent = metrics_agent.get('CC_delta', 0.0)
        cc_delta_std = metrics_std.get('CC_delta', 0.0)
        
        print(f"\n{'='*60}")
        print(f"COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"PSNR_psi  -> Agent: {score_agent:.2f} dB, Standard: {score_std:.2f} dB")
        print(f"CC_psi    -> Agent: {cc_psi_agent:.4f}, Standard: {cc_psi_std:.4f}")
        print(f"CC_delta  -> Agent: {cc_delta_agent:.4f}, Standard: {cc_delta_std:.4f}")
        print(f"{'='*60}")
        
        # Determine success (higher is better for PSNR and CC)
        # Allow 10% margin of error
        tolerance = 0.90
        
        psnr_pass = score_agent >= score_std * tolerance
        cc_psi_pass = cc_psi_agent >= cc_psi_std * tolerance
        cc_delta_pass = cc_delta_agent >= cc_delta_std * tolerance
        
        print(f"\n[CHECK] PSNR pass: {psnr_pass} (threshold: {score_std * tolerance:.2f})")
        print(f"[CHECK] CC_psi pass: {cc_psi_pass} (threshold: {cc_psi_std * tolerance:.4f})")
        print(f"[CHECK] CC_delta pass: {cc_delta_pass} (threshold: {cc_delta_std * tolerance:.4f})")
        
        # Overall pass if all metrics are acceptable
        all_pass = psnr_pass and cc_psi_pass and cc_delta_pass
        
        if all_pass:
            print("\n[SUCCESS] Agent's performance is acceptable!")
            sys.exit(0)
        else:
            print("\n[FAILURE] Agent's performance degraded significantly!")
            sys.exit(1)
            
    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()