import json

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np

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
