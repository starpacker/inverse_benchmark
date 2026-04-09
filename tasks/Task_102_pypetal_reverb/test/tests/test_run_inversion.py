import sys
import os
import dill
import numpy as np
import traceback
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the agent's function
from agent_run_inversion import run_inversion


def evaluate_results(data, inversion_result, results_dir, assets_dir):
    """
    Compute metrics, generate visualizations, and save outputs.
    
    Parameters:
    -----------
    data : dict
        Preprocessed data dictionary
    inversion_result : dict
        Results from run_inversion containing recovered transfer function
    results_dir : str
        Directory to save results
    assets_dir : str
        Directory to save assets
    
    Returns:
    --------
    metrics : dict
        Dictionary containing PSNR, CC, RMSE, and peak_lag_error
    """
    psi_gt = data['psi_gt']
    psi_rec = inversion_result['psi_rec']
    tau = data['tau']
    dt = data['dt']
    tau_peak = data['tau_peak']
    t = data['t']
    continuum = data['continuum']
    line_clean = data['line_clean']
    line_obs = data['line_obs']
    ccf_lags = inversion_result['ccf_lags']
    ccf = inversion_result['ccf']
    
    # Compute metrics
    max_lag = 80.0
    mask = tau <= max_lag
    gt = psi_gt[mask]
    rec = psi_rec[mask]
    
    # Normalise both to peak = 1 for scale-invariant comparison
    gt_peak = gt.max()
    rec_peak = rec.max()
    gt_norm = gt / gt_peak if gt_peak > 0 else gt
    rec_norm = rec / rec_peak if rec_peak > 0 else rec
    
    mse = np.mean((gt_norm - rec_norm)**2)
    psnr = 10.0 * np.log10(1.0 / mse) if mse > 0 else 100.0
    cc = float(np.corrcoef(gt_norm, rec_norm)[0, 1])
    rmse = float(np.sqrt(mse))
    
    # Peak lag recovery
    peak_gt = tau[mask][np.argmax(gt)]
    peak_rec = tau[mask][np.argmax(rec)]
    peak_error = abs(peak_rec - peak_gt)
    
    metrics = {
        "PSNR": float(psnr),
        "CC": float(cc),
        "RMSE": float(rmse),
        "peak_lag_error": float(peak_error),
    }
    
    print(f"    PSNR = {psnr:.2f} dB")
    print(f"    CC   = {cc:.4f}")
    print(f"    RMSE = {rmse:.6f}")
    print(f"    Peak lag error = {peak_error:.1f} days")
    
    # Save outputs
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    
    for d in [results_dir, assets_dir]:
        np.save(os.path.join(d, "gt_output.npy"), psi_gt)
        np.save(os.path.join(d, "recon_output.npy"), psi_rec)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    # Generate visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    ccf_mask = ccf_lags <= max_lag
    
    # Panel 1: Light curves
    ax = axes[0, 0]
    ax.plot(t, continuum, 'b-', lw=0.8, label='Continuum C(t)')
    ax.plot(t, line_obs, 'r-', lw=0.8, alpha=0.6, label='Line L(t) [observed]')
    ax.plot(t, line_clean, 'g--', lw=0.8, label='Line L(t) [clean]')
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Flux")
    ax.set_title("AGN Light Curves")
    ax.legend(fontsize=8)
    
    # Panel 2: Transfer function
    ax = axes[0, 1]
    ax.plot(tau[mask], psi_gt[mask], 'b-', lw=2, label='GT  Ψ(τ)')
    ax.plot(tau[mask], rec_norm, 'r--', lw=2, label='Recovered Ψ(τ)')
    ax.axvline(tau_peak, color='gray', ls=':', lw=1, label=f'True peak = {tau_peak} d')
    ax.set_xlabel("Lag τ (days)")
    ax.set_ylabel("Ψ(τ)")
    ax.set_title(f"Transfer Function | PSNR={metrics['PSNR']:.1f} dB, CC={metrics['CC']:.4f}")
    ax.legend(fontsize=8)
    
    # Panel 3: CCF
    ax = axes[1, 0]
    ax.plot(ccf_lags[ccf_mask], ccf[ccf_mask], 'k-', lw=1.5)
    ax.axvline(tau_peak, color='r', ls='--', lw=1, label=f'True lag = {tau_peak} d')
    peak_ccf_lag = ccf_lags[ccf_mask][np.argmax(ccf[ccf_mask])]
    ax.axvline(peak_ccf_lag, color='b', ls=':', lw=1, label=f'CCF peak = {peak_ccf_lag:.1f} d')
    ax.set_xlabel("Lag τ (days)")
    ax.set_ylabel("CCF")
    ax.set_title("Cross-Correlation Function")
    ax.legend(fontsize=8)
    
    # Panel 4: Residual of transfer function
    ax = axes[1, 1]
    residual = gt_norm - rec_norm
    ax.plot(tau[mask], residual, 'k-', lw=1)
    ax.axhline(0, color='r', ls='--', lw=0.5)
    ax.set_xlabel("Lag τ (days)")
    ax.set_ylabel("Residual")
    ax.set_title(f"Ψ Residual | Peak error = {metrics['peak_lag_error']:.1f} days")
    ax.fill_between(tau[mask], residual, alpha=0.3, color='gray')
    
    plt.tight_layout()
    for path in [os.path.join(results_dir, "vis_result.png"),
                 os.path.join(assets_dir, "vis_result.png")]:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    
    return metrics


def main():
    data_paths = ['/data/yjh/pypetal_reverb_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
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
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print(f"Outer data keys: {list(outer_data.keys())}")
    
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    # Run agent's function
    print("Running agent's run_inversion...")
    try:
        agent_output = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"ERROR: Agent's run_inversion failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine execution pattern
    if len(inner_paths) > 0:
        # Pattern 2: Chained execution
        print(f"Detected chained execution with {len(inner_paths)} inner file(s).")
        inner_path = inner_paths[0]
        print(f"Loading inner data from: {inner_path}")
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        print("Running chained call (agent_output as operator)...")
        try:
            final_result = agent_output(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR: Chained call failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Pattern 1: Direct execution
        print("Detected direct execution pattern.")
        final_result = agent_output
        std_result = std_output
    
    # The input data dict (first positional arg) contains ground truth info needed by evaluate_results
    input_data = args[0] if len(args) > 0 else kwargs.get('data', {})
    
    # Setup output directories
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_agent")
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets_agent")
    results_dir_std = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_std")
    assets_dir_std = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets_std")
    
    # Evaluate agent's result
    print("\n=== Evaluating Agent's Result ===")
    try:
        metrics_agent = evaluate_results(input_data, final_result, results_dir, assets_dir)
    except Exception as e:
        print(f"ERROR: Evaluation of agent result failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Evaluate standard result
    print("\n=== Evaluating Standard Result ===")
    try:
        metrics_std = evaluate_results(input_data, std_result, results_dir_std, assets_dir_std)
    except Exception as e:
        print(f"ERROR: Evaluation of standard result failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Compare metrics
    print("\n=== Comparison ===")
    print(f"  Agent  PSNR: {metrics_agent['PSNR']:.2f} dB, CC: {metrics_agent['CC']:.4f}, RMSE: {metrics_agent['RMSE']:.6f}, Peak Lag Error: {metrics_agent['peak_lag_error']:.1f}")
    print(f"  Standard PSNR: {metrics_std['PSNR']:.2f} dB, CC: {metrics_std['CC']:.4f}, RMSE: {metrics_std['RMSE']:.6f}, Peak Lag Error: {metrics_std['peak_lag_error']:.1f}")
    
    score_agent = metrics_agent['PSNR']
    score_std = metrics_std['PSNR']
    
    print(f"\nScores -> Agent: {score_agent}, Standard: {score_std}")
    
    # PSNR: Higher is better. Allow 10% margin.
    passed = True
    
    # Check PSNR (higher is better)
    if score_std > 0:
        if score_agent < score_std * 0.9:
            print(f"FAIL: Agent PSNR ({score_agent:.2f}) is significantly lower than Standard ({score_std:.2f})")
            passed = False
    elif score_std <= 0:
        # Both could be negative for very bad reconstructions; allow some margin
        if score_agent < score_std - abs(score_std) * 0.1:
            print(f"FAIL: Agent PSNR ({score_agent:.2f}) is significantly lower than Standard ({score_std:.2f})")
            passed = False
    
    # Check CC (higher is better, range [-1, 1])
    cc_agent = metrics_agent['CC']
    cc_std = metrics_std['CC']
    if cc_agent < cc_std - 0.05:
        print(f"FAIL: Agent CC ({cc_agent:.4f}) is significantly lower than Standard ({cc_std:.4f})")
        passed = False
    
    # Check RMSE (lower is better)
    rmse_agent = metrics_agent['RMSE']
    rmse_std = metrics_std['RMSE']
    if rmse_std > 0:
        if rmse_agent > rmse_std * 1.1:
            print(f"FAIL: Agent RMSE ({rmse_agent:.6f}) is significantly higher than Standard ({rmse_std:.6f})")
            passed = False
    
    # Check peak lag error (lower is better)
    ple_agent = metrics_agent['peak_lag_error']
    ple_std = metrics_std['peak_lag_error']
    if ple_agent > ple_std + 5.0:  # Allow 5 days extra tolerance
        print(f"FAIL: Agent peak lag error ({ple_agent:.1f}) is significantly higher than Standard ({ple_std:.1f})")
        passed = False
    
    if passed:
        print("\nRESULT: PASS - Agent performance is within acceptable range.")
        sys.exit(0)
    else:
        print("\nRESULT: FAIL - Agent performance degraded significantly.")
        sys.exit(1)


if __name__ == "__main__":
    main()