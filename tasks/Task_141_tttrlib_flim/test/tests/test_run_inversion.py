import sys
import os
import dill
import numpy as np
import traceback
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the target function
from agent_run_inversion import run_inversion

# ============================================================
# Inject the referee evaluation function verbatim
# ============================================================
def evaluate_results(
    data: dict,
    inversion_result: dict,
    results_dir: str
) -> dict:
    """
    Evaluate inversion results and generate outputs.
    
    Computes metrics (PSNR, correlation, relative errors), saves results,
    and generates visualization.
    
    Parameters
    ----------
    data : dict
        Data dictionary from load_and_preprocess_data.
    inversion_result : dict
        Results dictionary from run_inversion.
    results_dir : str
        Directory to save outputs.
        
    Returns
    -------
    dict
        Dictionary of computed metrics.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    time = data['time']
    gt_curve = data['gt_curve']
    gt_curve_with_bg = data['gt_curve_with_bg']
    measured = data['measured']
    irf = data['irf']
    params = data['params']
    
    a1_fit = inversion_result['a1_fit']
    a2_fit = inversion_result['a2_fit']
    tau1_fit = inversion_result['tau1_fit']
    tau2_fit = inversion_result['tau2_fit']
    bg_fit = inversion_result['bg_fit']
    fitted_curve = inversion_result['fitted_curve']
    reduced_chi2 = inversion_result['reduced_chi2']
    
    tau1_true = params['tau1_true']
    tau2_true = params['tau2_true']
    a1_true = params['a1_true']
    a2_true = params['a2_true']
    background = params['background']
    
    # Compare fitted curve (without bg) to ground-truth curve (without bg)
    gt_no_bg = gt_curve
    fitted_no_bg = fitted_curve - bg_fit
    
    # PSNR
    mse = np.mean((gt_no_bg - fitted_no_bg) ** 2)
    if mse == 0:
        psnr_val = float("inf")
    else:
        peak = np.max(gt_no_bg)
        psnr_val = 10.0 * np.log10(peak ** 2 / mse)
    
    # Correlation coefficient
    a_c = gt_no_bg - gt_no_bg.mean()
    b_c = fitted_no_bg - fitted_no_bg.mean()
    cc_val = float(np.sum(a_c * b_c) / (np.sqrt(np.sum(a_c ** 2) * np.sum(b_c ** 2)) + 1e-30))
    
    # Relative errors
    def relative_error(true_val, fit_val):
        return abs(fit_val - true_val) / abs(true_val)
    
    re_tau1 = relative_error(tau1_true, tau1_fit)
    re_tau2 = relative_error(tau2_true, tau2_fit)
    re_a1 = relative_error(a1_true, a1_fit)
    re_a2 = relative_error(a2_true, a2_fit)
    
    metrics = {
        "PSNR_dB": round(float(psnr_val), 2),
        "CC": round(float(cc_val), 6),
        "reduced_chi2": round(float(reduced_chi2), 6),
        "tau1_true_ns": tau1_true,
        "tau1_fit_ns": round(float(tau1_fit), 4),
        "tau1_RE": round(float(re_tau1), 6),
        "tau2_true_ns": tau2_true,
        "tau2_fit_ns": round(float(tau2_fit), 4),
        "tau2_RE": round(float(re_tau2), 6),
        "a1_true": a1_true,
        "a1_fit": round(float(a1_fit), 4),
        "a1_RE": round(float(re_a1), 6),
        "a2_true": a2_true,
        "a2_fit": round(float(a2_fit), 4),
        "a2_RE": round(float(re_a2), 6),
        "background_true": background,
        "background_fit": round(float(bg_fit), 2),
    }
    
    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    # Save outputs
    np.save(os.path.join(results_dir, "ground_truth.npy"), gt_no_bg)
    np.save(os.path.join(results_dir, "recon_output.npy"), fitted_no_bg)
    
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\nSaved ground_truth.npy, recon_output.npy, metrics.json")
    
    # Visualization - 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Task 141: TCSPC Fluorescence Decay Deconvolution", fontsize=14, fontweight="bold")
    
    # Panel 1: Decay + Fit (log scale)
    ax = axes[0, 0]
    ax.semilogy(time, measured, "k.", markersize=1.5, alpha=0.5, label="Measured (Poisson)")
    ax.semilogy(time, gt_curve_with_bg, "b-", linewidth=1.5, label="Ground truth + bg")
    ax.semilogy(time, fitted_curve, "r--", linewidth=1.5, label="Fitted model")
    ax.semilogy(time, irf * irf.max() ** -1 * gt_curve_with_bg.max() * 0.3,
                "g-", linewidth=1, alpha=0.6, label="IRF (scaled)")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Counts")
    ax.set_title("Decay Curve & Fit")
    ax.legend(fontsize=8)
    ax.set_xlim(0, time[-1])
    ax.set_ylim(bottom=1)
    
    # Panel 2: Weighted residuals
    ax = axes[0, 1]
    residuals = (measured - fitted_curve) / np.sqrt(np.maximum(fitted_curve, 1.0))
    ax.plot(time, residuals, "k-", linewidth=0.5, alpha=0.7)
    ax.axhline(0, color="r", linestyle="--", linewidth=0.8)
    ax.axhline(2, color="gray", linestyle=":", linewidth=0.5)
    ax.axhline(-2, color="gray", linestyle=":", linewidth=0.5)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Weighted Residual")
    ax.set_title(f"Residuals  (χ²ᵣ = {reduced_chi2:.4f})")
    ax.set_xlim(0, time[-1])
    
    # Panel 3: Parameter comparison (bar chart)
    ax = axes[1, 0]
    param_names = ["τ₁ (ns)", "τ₂ (ns)", "a₁", "a₂", "bg"]
    true_vals = [tau1_true, tau2_true, a1_true, a2_true, background]
    fit_vals = [tau1_fit, tau2_fit, a1_fit, a2_fit, bg_fit]
    x_pos = np.arange(len(param_names))
    width = 0.35
    bars1 = ax.bar(x_pos - width / 2, true_vals, width, label="True", color="steelblue", alpha=0.8)
    bars2 = ax.bar(x_pos + width / 2, fit_vals, width, label="Fitted", color="coral", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(param_names)
    ax.set_ylabel("Value")
    ax.set_title("Parameter Recovery")
    ax.legend()
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    
    # Panel 4: Individual decay components
    ax = axes[1, 1]
    comp1_true = a1_true * np.exp(-time / tau1_true)
    comp2_true = a2_true * np.exp(-time / tau2_true)
    comp1_fit = a1_fit * np.exp(-time / tau1_fit)
    comp2_fit = a2_fit * np.exp(-time / tau2_fit)
    ax.semilogy(time, comp1_true, "b-", linewidth=1.5, label=f"True τ₁={tau1_true} ns")
    ax.semilogy(time, comp1_fit, "b--", linewidth=1.5, label=f"Fit τ₁={tau1_fit:.3f} ns")
    ax.semilogy(time, comp2_true, "r-", linewidth=1.5, label=f"True τ₂={tau2_true} ns")
    ax.semilogy(time, comp2_fit, "r--", linewidth=1.5, label=f"Fit τ₂={tau2_fit:.3f} ns")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Decay Components")
    ax.legend(fontsize=8)
    ax.set_xlim(0, time[-1])
    
    plt.tight_layout()
    fig_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path}")
    
    return metrics


# ============================================================
# Main test logic
# ============================================================
def main():
    data_paths = ['/data/yjh/tttrlib_flim_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Classify files
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p
    
    if outer_path is None:
        print("ERROR: No outer (primary) data file found.")
        sys.exit(1)
    
    # Load outer data
    print(f"Loading outer data from: {outer_path}")
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)
    
    print(f"Outer data keys: {list(outer_data.keys())}")
    print(f"Function name: {outer_data.get('func_name', 'unknown')}")
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    # Run the agent's function
    print("\n=== Running agent's run_inversion ===")
    try:
        agent_output = run_inversion(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Agent function failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check for chained execution
    if len(inner_paths) > 0:
        print(f"\nChained execution detected. Inner files: {inner_paths}")
        # Load inner data
        inner_path = inner_paths[0]
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        # agent_output should be callable
        try:
            final_result = agent_output(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR: Inner call failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Direct execution
        final_result = agent_output
        std_result = std_output
    
    print("\n=== Agent output keys ===")
    if isinstance(final_result, dict):
        print(list(final_result.keys()))
    print("\n=== Standard output keys ===")
    if isinstance(std_result, dict):
        print(list(std_result.keys()))
    
    # We need a 'data' dict for evaluate_results. 
    # This must contain: time, gt_curve, gt_curve_with_bg, measured, irf, params
    # These are the inputs to run_inversion (time, irf, measured) plus ground truth info.
    # The outer_args are: (time, irf, measured, total_counts, bounds, rng_seed, ...)
    # We need to reconstruct the 'data' dict. Let's extract what we can from outer_args.
    
    # Extract inputs from outer_args
    # run_inversion signature: time, irf, measured, total_counts, bounds, rng_seed, maxiter, tol
    time_arr = outer_args[0] if len(outer_args) > 0 else outer_kwargs.get('time')
    irf_arr = outer_args[1] if len(outer_args) > 1 else outer_kwargs.get('irf')
    measured_arr = outer_args[2] if len(outer_args) > 2 else outer_kwargs.get('measured')
    total_counts = outer_args[3] if len(outer_args) > 3 else outer_kwargs.get('total_counts')
    
    # For evaluate_results we need gt_curve, gt_curve_with_bg, and params with true values.
    # We'll use the standard result to reconstruct what we can, or use the fitted values from std.
    # Since we're comparing agent vs standard, we can construct a synthetic 'data' dict.
    # 
    # The key insight: we use the SAME data dict for both evaluations, so the comparison is fair.
    # We'll use the standard result's fitted values as "ground truth" if actual GT is not available.
    # But actually, we should try to find if the pkl has any additional info.
    
    # Let's check if there's additional data stored
    print(f"\nOuter data all keys: {list(outer_data.keys())}")
    
    # We need to construct the data dict for evaluate_results.
    # Since we don't have explicit ground truth in the pkl, we'll use the standard output
    # to derive ground truth parameters and curves.
    # 
    # The standard result has: a1_fit, a2_fit, tau1_fit, tau2_fit, bg_fit, fitted_curve, reduced_chi2, success
    # We'll treat the standard's fitted parameters as "true" parameters for evaluation purposes,
    # and compute gt_curve from them.
    
    from scipy.signal import fftconvolve
    
    def make_forward(time, irf, a1, tau1, a2, tau2, bg, total_counts):
        decay = a1 * np.exp(-time / tau1) + a2 * np.exp(-time / tau2)
        convolved = fftconvolve(irf, decay, mode='full')[:len(time)]
        convolved = convolved / convolved.sum() * total_counts
        return convolved  # without bg
    
    # Use standard result as ground truth
    std_a1 = std_result['a1_fit']
    std_a2 = std_result['a2_fit']
    std_tau1 = std_result['tau1_fit']
    std_tau2 = std_result['tau2_fit']
    std_bg = std_result['bg_fit']
    
    gt_curve_no_bg = make_forward(time_arr, irf_arr, std_a1, std_tau1, std_a2, std_tau2, std_bg, total_counts)
    gt_curve_with_bg = gt_curve_no_bg + std_bg
    
    data_dict = {
        'time': time_arr,
        'gt_curve': gt_curve_no_bg,
        'gt_curve_with_bg': gt_curve_with_bg,
        'measured': measured_arr,
        'irf': irf_arr,
        'params': {
            'tau1_true': std_tau1,
            'tau2_true': std_tau2,
            'a1_true': std_a1,
            'a2_true': std_a2,
            'background': std_bg,
        }
    }
    
    # Evaluate agent result
    print("\n" + "=" * 60)
    print("Evaluating AGENT result...")
    print("=" * 60)
    agent_results_dir = os.path.join(os.getcwd(), "results_agent")
    try:
        agent_metrics = evaluate_results(data_dict, final_result, agent_results_dir)
    except Exception as e:
        print(f"ERROR evaluating agent results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Evaluate standard result
    print("\n" + "=" * 60)
    print("Evaluating STANDARD result...")
    print("=" * 60)
    std_results_dir = os.path.join(os.getcwd(), "results_standard")
    try:
        std_metrics = evaluate_results(data_dict, std_result, std_results_dir)
    except Exception as e:
        print(f"ERROR evaluating standard results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract primary metrics for comparison
    agent_psnr = agent_metrics['PSNR_dB']
    std_psnr = std_metrics['PSNR_dB']
    agent_cc = agent_metrics['CC']
    std_cc = std_metrics['CC']
    agent_chi2 = agent_metrics['reduced_chi2']
    std_chi2 = std_metrics['reduced_chi2']
    
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"PSNR (dB)     -> Agent: {agent_psnr}, Standard: {std_psnr}")
    print(f"CC            -> Agent: {agent_cc}, Standard: {std_cc}")
    print(f"Reduced Chi2  -> Agent: {agent_chi2}, Standard: {std_chi2}")
    
    # Also compare fitted parameters directly
    print("\nParameter comparison:")
    for key in ['a1_fit', 'a2_fit', 'tau1_fit', 'tau2_fit', 'bg_fit']:
        agent_val = final_result[key]
        std_val = std_result[key]
        if abs(std_val) > 1e-10:
            rel_diff = abs(agent_val - std_val) / abs(std_val) * 100
        else:
            rel_diff = abs(agent_val - std_val) * 100
        print(f"  {key}: Agent={agent_val:.6f}, Std={std_val:.6f}, RelDiff={rel_diff:.4f}%")
    
    # Determine success
    # PSNR: higher is better. We allow some margin.
    # CC: higher is better (closer to 1).
    # reduced_chi2: closer to 1 is ideal, but we compare relative to standard.
    
    success = True
    margin = 0.10  # 10% margin
    
    # PSNR check: agent should be at least 90% of standard (or better)
    if std_psnr != float('inf'):
        if agent_psnr < std_psnr * (1 - margin) and std_psnr > 0:
            print(f"\nWARNING: Agent PSNR ({agent_psnr}) is significantly lower than Standard ({std_psnr})")
            success = False
    
    # CC check: agent should be close to standard
    if agent_cc < std_cc * (1 - margin) and std_cc > 0:
        print(f"\nWARNING: Agent CC ({agent_cc}) is significantly lower than Standard ({std_cc})")
        success = False
    
    # Check that agent's optimization converged
    if not final_result.get('success', False):
        print("\nWARNING: Agent optimization did not converge (success=False)")
        # Don't fail on this alone if metrics are good
    
    # Check parameter recovery (relative errors should be reasonable)
    for key in ['tau1_RE', 'tau2_RE', 'a1_RE', 'a2_RE']:
        agent_re = agent_metrics.get(key, 999)
        std_re = std_metrics.get(key, 999)
        # Agent's relative error should not be dramatically worse
        if agent_re > 0.5 and agent_re > std_re * 5:
            print(f"\nWARNING: Agent {key} ({agent_re}) is much worse than Standard ({std_re})")
            success = False
    
    if success:
        print("\n✅ TEST PASSED: Agent performance is acceptable.")
        sys.exit(0)
    else:
        print("\n❌ TEST FAILED: Agent performance degraded significantly.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)