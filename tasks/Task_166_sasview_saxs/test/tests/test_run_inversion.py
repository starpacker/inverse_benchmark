import sys
import os
import dill
import numpy as np
import traceback
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import the target function
from agent_run_inversion import run_inversion

# ============================================================================
# INJECT REFEREE (Evaluation Function) - Copied verbatim from Reference B
# ============================================================================

def compute_pr(q, I_q, d_max=None, n_r=100):
    """
    Estimate the pair-distance distribution function P(r) via a simple
    indirect Fourier transform (Moore method / regularised sine transform).
    
    P(r) = (2r / pi) * integral_0^inf  q * I(q) * sin(qr) dq
    
    In practice we discretise and apply a simple Tikhonov regularisation
    to suppress noise artefacts.
    """
    if d_max is None:
        d_max = 2.0 * np.pi / q.min()
        d_max = min(d_max, 300.0)

    r = np.linspace(0, d_max, n_r)
    pr = np.zeros_like(r)

    for i, ri in enumerate(r):
        if ri < 1e-12:
            pr[i] = 0.0
            continue
        integrand = q * I_q * np.sin(q * ri)
        pr[i] = (2.0 * ri / np.pi) * np.trapezoid(integrand, q)

    pr = np.maximum(pr, 0.0)
    if pr.max() > 0:
        pr /= pr.max()

    return r, pr

def evaluate_results(data, inversion_result, results_dir):
    """
    Evaluate results: Compute metrics and generate visualizations.
    
    Parameters:
    -----------
    data : dict
        Preprocessed data from load_and_preprocess_data
    inversion_result : dict
        Results from run_inversion
    results_dir : str
        Directory to save results
        
    Returns:
    --------
    dict containing all evaluation metrics
    """
    print("[EVAL] Computing evaluation metrics ...")
    
    os.makedirs(results_dir, exist_ok=True)
    
    q = data['q']
    I_clean = data['I_clean']
    I_noisy = data['I_noisy']
    sigma = data['sigma']
    gt_params = data['gt_params']
    
    fitted_params = inversion_result['fitted_params']
    I_fit = inversion_result['I_fit']
    r_pr = inversion_result['r_pr']
    pr_fitted = inversion_result['P_r']
    fit_success = inversion_result['fit_success']
    
    GT_RADIUS = gt_params['radius']
    GT_SCALE = gt_params['scale']
    GT_BACKGROUND = gt_params['background']
    
    R_fit = fitted_params['radius']
    scale_fit = fitted_params['scale']
    bg_fit = fitted_params['background']
    
    RE_R = abs(R_fit - GT_RADIUS) / GT_RADIUS * 100
    RE_scale = abs(scale_fit - GT_SCALE) / GT_SCALE * 100
    RE_bg = abs(bg_fit - GT_BACKGROUND) / GT_BACKGROUND * 100
    
    print(f"[EVAL] Relative Error R:     {RE_R:.4f}%")
    print(f"[EVAL] Relative Error scale: {RE_scale:.4f}%")
    print(f"[EVAL] Relative Error bg:    {RE_bg:.4f}%")
    
    residuals = I_noisy - I_fit
    mse_noisy = np.mean(residuals**2)
    max_I = np.max(I_noisy)
    psnr_noisy = 10.0 * np.log10(max_I**2 / mse_noisy) if mse_noisy > 0 else float('inf')
    cc_noisy = np.corrcoef(I_noisy, I_fit)[0, 1]
    
    residuals_clean = I_clean - I_fit
    mse_clean = np.mean(residuals_clean**2)
    psnr_clean = 10.0 * np.log10(np.max(I_clean)**2 / mse_clean) if mse_clean > 0 else float('inf')
    cc_clean = np.corrcoef(I_clean, I_fit)[0, 1]
    
    psnr = psnr_clean
    cc = cc_clean
    
    log_I_clean = np.log10(np.maximum(I_clean, 1e-10))
    log_I_fit = np.log10(np.maximum(I_fit, 1e-10))
    rmse_log = np.sqrt(np.mean((log_I_clean - log_I_fit)**2))
    
    print(f"[EVAL] I(q) PSNR (vs GT):    {psnr:.2f} dB")
    print(f"[EVAL] I(q) CC (vs GT):      {cc:.6f}")
    print(f"[EVAL] I(q) PSNR (vs noisy): {psnr_noisy:.2f} dB")
    print(f"[EVAL] I(q) CC (vs noisy):   {cc_noisy:.6f}")
    print(f"[EVAL] I(q) RMSE_log:        {rmse_log:.6f}")
    
    print("[VIS] Generating visualization ...")
    
    I_gt_for_pr = I_clean - GT_BACKGROUND
    I_gt_for_pr = np.maximum(I_gt_for_pr, 1e-10)
    r_gt, pr_gt = compute_pr(q, I_gt_for_pr, d_max=2.5 * GT_RADIUS * 1.5, n_r=150)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    ax = axes[0, 0]
    ax.loglog(q, I_noisy, 'o', ms=2, alpha=0.5, color='steelblue', label='Measured (noisy)')
    ax.loglog(q, I_clean, '-', lw=1.5, color='black', alpha=0.6, label='Ground Truth')
    ax.loglog(q, I_fit, '-', lw=2, color='red', label='Fitted')
    ax.set_xlabel(r'$q$ (Å$^{-1}$)', fontsize=12)
    ax.set_ylabel(r'$I(q)$ (a.u.)', fontsize=12)
    ax.set_title('(a) Scattering Intensity I(q)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    ax = axes[0, 1]
    residuals_plot = I_noisy - I_fit
    ax.semilogx(q, residuals_plot, '-', lw=0.8, color='navy', alpha=0.7)
    ax.axhline(0, color='red', ls='--', lw=1)
    ax.fill_between(q, -2*sigma, 2*sigma, alpha=0.15, color='orange', label=r'$\pm 2\sigma$')
    ax.set_xlabel(r'$q$ (Å$^{-1}$)', fontsize=12)
    ax.set_ylabel(r'$I_{meas} - I_{fit}$', fontsize=12)
    ax.set_title('(b) Fit Residuals', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    param_names = ['Radius (Å)', 'Scale (×1e3)', 'Background (×1e3)']
    gt_vals = [GT_RADIUS, GT_SCALE * 1e3, GT_BACKGROUND * 1e3]
    fit_vals = [R_fit, scale_fit * 1e3, bg_fit * 1e3]
    x_pos = np.arange(len(param_names))
    width = 0.35
    ax.bar(x_pos - width/2, gt_vals, width, label='Ground Truth', color='steelblue', alpha=0.8)
    ax.bar(x_pos + width/2, fit_vals, width, label='Fitted', color='coral', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(param_names, fontsize=10)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('(c) Parameter Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (g, f) in enumerate(zip(gt_vals, fit_vals)):
        re = abs(f - g) / g * 100
        ax.annotate(f'RE={re:.2f}%', xy=(x_pos[i], max(g, f)*1.05),
                    ha='center', fontsize=9, color='darkred', fontweight='bold')
    
    ax = axes[1, 1]
    ax.plot(r_gt, pr_gt, '-', lw=2, color='black', label='P(r) from GT I(q)')
    ax.plot(r_pr, pr_fitted, '--', lw=2, color='red', label='P(r) from fitted I(q)')
    ax.axvline(2*GT_RADIUS, color='blue', ls=':', lw=1.5, alpha=0.7, label=f'D_max=2R={2*GT_RADIUS} Å')
    ax.set_xlabel(r'$r$ (Å)', fontsize=12)
    ax.set_ylabel(r'$P(r)$ (normalised)', fontsize=12)
    ax.set_title('(d) Pair-Distance Distribution P(r)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)
    
    plt.suptitle(f'SAXS Sphere Model Fitting — R={R_fit:.2f} Å (GT={GT_RADIUS}), '
                 f'PSNR={psnr:.1f} dB, CC={cc:.4f}',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    fig_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved figure to {fig_path}")
    
    print("[SAVE] Saving outputs ...")
    
    metrics = {
        "radius_gt": GT_RADIUS,
        "radius_fitted": float(R_fit),
        "radius_RE_percent": float(RE_R),
        "scale_gt": GT_SCALE,
        "scale_fitted": float(scale_fit),
        "scale_RE_percent": float(RE_scale),
        "background_gt": GT_BACKGROUND,
        "background_fitted": float(bg_fit),
        "background_RE_percent": float(RE_bg),
        "Iq_PSNR_dB_vs_GT": float(psnr_clean),
        "Iq_CC_vs_GT": float(cc_clean),
        "Iq_PSNR_dB_vs_noisy": float(psnr_noisy),
        "Iq_CC_vs_noisy": float(cc_noisy),
        "Iq_RMSE_log": float(rmse_log),
        "PSNR": float(psnr),
        "CC": float(cc),
        "fit_success": fit_success
    }
    
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Saved metrics to {metrics_path}")
    
    gt_data = {
        "q": q,
        "I_q_clean": I_clean,
        "I_q_noisy": I_noisy,
        "sigma": sigma,
        "parameters": {
            "radius": GT_RADIUS,
            "scale": GT_SCALE,
            "background": GT_BACKGROUND
        }
    }
    gt_path = os.path.join(results_dir, "ground_truth.npy")
    np.save(gt_path, gt_data, allow_pickle=True)
    print(f"[SAVE] Saved ground truth to {gt_path}")
    
    recon_data = {
        "q": q,
        "I_q_fitted": I_fit,
        "I_q_noisy": I_noisy,
        "r_pr": r_pr,
        "P_r": pr_fitted,
        "parameters": {
            "radius": float(R_fit),
            "scale": float(scale_fit),
            "background": float(bg_fit)
        }
    }
    recon_path = os.path.join(results_dir, "reconstruction.npy")
    np.save(recon_path, recon_data, allow_pickle=True)
    print(f"[SAVE] Saved reconstruction to {recon_path}")
    
    print("\n" + "="*60)
    print("[DONE] SAXS sphere model fitting complete!")
    print(f"  Radius: {R_fit:.4f} Å  (GT={GT_RADIUS}, RE={RE_R:.4f}%)")
    print(f"  Scale:  {scale_fit:.6f}  (GT={GT_SCALE}, RE={RE_scale:.4f}%)")
    print(f"  BG:     {bg_fit:.6f}  (GT={GT_BACKGROUND}, RE={RE_bg:.4f}%)")
    print(f"  PSNR:   {psnr:.2f} dB")
    print(f"  CC:     {cc:.6f}")
    print("="*60)
    
    return metrics

# ============================================================================
# Main Test Logic
# ============================================================================

def main():
    # Data paths provided
    data_paths = ['/data/yjh/sasview_saxs_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    print("=" * 70)
    print("TEST: run_inversion Performance Validation")
    print("=" * 70)
    
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
    
    # We expect at least one outer data file
    if not outer_data_files:
        print("ERROR: No outer data file found!")
        sys.exit(1)
    
    # Load the primary (outer) data
    outer_data_path = outer_data_files[0]
    print(f"\nLoading outer data from: {outer_data_path}")
    
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print(f"Outer data keys: {outer_data.keys()}")
    
    # Extract args, kwargs, and expected output
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    print(f"Args length: {len(args)}")
    print(f"Kwargs keys: {kwargs.keys()}")
    
    # Run the agent's run_inversion function
    print("\n" + "-" * 50)
    print("Running Agent's run_inversion...")
    print("-" * 50)
    
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"ERROR running agent's run_inversion: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("Agent's run_inversion completed successfully.")
    
    # Check if this is a chained execution (closure/factory pattern)
    if inner_data_files and callable(agent_output):
        # Chained execution pattern
        print("\nDetected chained execution pattern (closure/factory).")
        inner_data_path = inner_data_files[0]
        print(f"Loading inner data from: {inner_data_path}")
        
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
        
        print("Running operator from agent_output with inner data...")
        try:
            final_result = agent_output(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR running inner function: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Direct execution pattern
        print("\nDirect execution pattern.")
        final_result = agent_output
        std_result = std_output
    
    # Now we need to evaluate both results
    # The evaluate_results function requires 'data' (preprocessed data)
    # which should be the first argument passed to run_inversion
    
    # Extract the 'data' input from args
    if len(args) > 0:
        input_data = args[0]
    else:
        input_data = kwargs.get('data', None)
    
    if input_data is None:
        print("ERROR: Could not extract input data for evaluation!")
        sys.exit(1)
    
    print("\n" + "-" * 50)
    print("Evaluating Agent's Result...")
    print("-" * 50)
    
    # Create results directories
    agent_results_dir = './test_results_agent'
    std_results_dir = './test_results_std'
    
    try:
        agent_metrics = evaluate_results(input_data, final_result, agent_results_dir)
    except Exception as e:
        print(f"ERROR evaluating agent's result: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "-" * 50)
    print("Evaluating Standard (Ground Truth) Result...")
    print("-" * 50)
    
    try:
        std_metrics = evaluate_results(input_data, std_result, std_results_dir)
    except Exception as e:
        print(f"ERROR evaluating standard result: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract key metrics for comparison
    # Primary metrics: PSNR (higher is better), CC (higher is better)
    agent_psnr = agent_metrics.get('PSNR', 0)
    std_psnr = std_metrics.get('PSNR', 0)
    
    agent_cc = agent_metrics.get('CC', 0)
    std_cc = std_metrics.get('CC', 0)
    
    agent_radius_re = agent_metrics.get('radius_RE_percent', float('inf'))
    std_radius_re = std_metrics.get('radius_RE_percent', float('inf'))
    
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Metric':<30} {'Agent':<20} {'Standard':<20}")
    print("-" * 70)
    print(f"{'PSNR (dB)':<30} {agent_psnr:<20.4f} {std_psnr:<20.4f}")
    print(f"{'CC':<30} {agent_cc:<20.6f} {std_cc:<20.6f}")
    print(f"{'Radius RE (%)':<30} {agent_radius_re:<20.4f} {std_radius_re:<20.4f}")
    print(f"{'Fit Success':<30} {agent_metrics.get('fit_success', False):<20} {std_metrics.get('fit_success', False):<20}")
    print("=" * 70)
    
    # Verification: Allow 10% degradation margin
    # PSNR: Higher is better
    # CC: Higher is better (closer to 1)
    # Radius RE: Lower is better
    
    margin = 0.10  # 10% margin
    
    # Check if agent's fit was successful
    if not agent_metrics.get('fit_success', False):
        print("\nFAILED: Agent's fit was not successful!")
        sys.exit(1)
    
    # Check PSNR
    psnr_threshold = std_psnr * (1 - margin) if std_psnr > 0 else -float('inf')
    psnr_pass = agent_psnr >= psnr_threshold
    
    # Check CC
    cc_threshold = std_cc * (1 - margin) if std_cc > 0 else -float('inf')
    cc_pass = agent_cc >= cc_threshold
    
    # Check Radius Relative Error (lower is better, allow 50% more error)
    radius_re_threshold = std_radius_re * (1 + 0.5)  # Allow 50% more relative error
    radius_re_pass = agent_radius_re <= radius_re_threshold or agent_radius_re < 5.0  # Or if RE < 5%, it's good
    
    print(f"\nPSNR Check: Agent={agent_psnr:.4f} >= Threshold={psnr_threshold:.4f} -> {'PASS' if psnr_pass else 'FAIL'}")
    print(f"CC Check: Agent={agent_cc:.6f} >= Threshold={cc_threshold:.6f} -> {'PASS' if cc_pass else 'FAIL'}")
    print(f"Radius RE Check: Agent={agent_radius_re:.4f}% <= Threshold={radius_re_threshold:.4f}% -> {'PASS' if radius_re_pass else 'FAIL'}")
    
    # Overall pass if all critical metrics pass
    all_pass = psnr_pass and cc_pass and radius_re_pass
    
    print("\n" + "=" * 70)
    if all_pass:
        print("OVERALL RESULT: PASS")
        print("Agent's performance is acceptable.")
        print("=" * 70)
        sys.exit(0)
    else:
        print("OVERALL RESULT: FAIL")
        print("Agent's performance degraded significantly.")
        print("=" * 70)
        sys.exit(1)

if __name__ == "__main__":
    main()