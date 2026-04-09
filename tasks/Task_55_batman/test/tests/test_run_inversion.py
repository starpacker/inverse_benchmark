import sys
import os
import dill
import numpy as np
import traceback

# Import target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from skimage.metrics import structural_similarity as ssim_fn

# Inject the referee evaluation function
def evaluate_results(gt_params, fit_params, flux_clean, flux_fit, flux_meas,
                     t, results_dir):
    """
    Compute metrics, save results, and generate visualizations.
    
    Parameters
    ----------
    gt_params : dict
        Ground-truth parameters.
    fit_params : dict
        Fitted parameters.
    flux_clean : np.ndarray
        Ground-truth flux.
    flux_fit : np.ndarray
        Fitted flux.
    flux_meas : np.ndarray
        Measured (noisy) flux.
    t : np.ndarray
        Time array.
    results_dir : str
        Directory to save results.
    
    Returns
    -------
    metrics : dict
        Dictionary of computed metrics.
    """
    print("\n[EVAL] Computing metrics ...")
    
    # Light-curve metrics
    residual = flux_clean - flux_fit
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    cc = float(np.corrcoef(flux_clean, flux_fit)[0, 1])

    data_range = flux_clean.max() - flux_clean.min()
    mse = np.mean(residual ** 2)
    psnr = float(10 * np.log10(data_range ** 2 / max(mse, 1e-30)))

    tile_rows = 7
    a2d = np.tile(flux_clean, (tile_rows, 1))
    b2d = np.tile(flux_fit, (tile_rows, 1))
    ssim = float(ssim_fn(
        a2d, b2d,
        data_range=data_range, win_size=7
    ))

    # Relative error
    norm_gt = np.linalg.norm(flux_clean)
    re = float(np.linalg.norm(residual) / max(norm_gt, 1e-12))

    # Parameter recovery
    free_keys = ["rp", "a", "inc", "u1", "u2"]
    param_metrics = {}
    for k in free_keys:
        gt_v = gt_params[k]
        fit_v = fit_params[k]
        param_metrics[f"gt_{k}"] = float(gt_v)
        param_metrics[f"fit_{k}"] = float(fit_v)
        param_metrics[f"abs_err_{k}"] = float(abs(gt_v - fit_v))
        if abs(gt_v) > 1e-12:
            param_metrics[f"rel_err_{k}_pct"] = float(abs(gt_v - fit_v) / abs(gt_v) * 100)

    metrics = {
        "PSNR": psnr,
        "SSIM": ssim,
        "CC": cc,
        "RMSE": rmse,
        "RE": re,
        **param_metrics,
    }
    
    # Print metrics
    for k, v in sorted(metrics.items()):
        print(f"  {k:30s} = {v}")
    
    # Save results
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(results_dir, "reconstruction.npy"), flux_fit)
    np.save(os.path.join(results_dir, "ground_truth.npy"), flux_clean)
    np.save(os.path.join(results_dir, "measurements.npy"), flux_meas)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (a) Light curves
    ax = axes[0, 0]
    ax.plot(t * 24, flux_meas, 'k.', ms=1, alpha=0.3, label='Noisy data')
    ax.plot(t * 24, flux_clean, 'b-', lw=2, label='Ground truth')
    ax.plot(t * 24, flux_fit, 'r--', lw=1.5, label='batman fit')
    ax.set_xlabel('Time from mid-transit [hours]')
    ax.set_ylabel('Relative flux')
    ax.set_title('(a) Transit Light Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Residuals
    ax = axes[0, 1]
    residual_ppm = (flux_clean - flux_fit) * 1e6
    ax.plot(t * 24, residual_ppm, 'g-', lw=0.8)
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Residual [ppm]')
    ax.set_title(f'(b) Residuals  RMSE={metrics["RMSE"]*1e6:.1f} ppm')
    ax.grid(True, alpha=0.3)

    # (c) Transit depth zoom
    ax = axes[1, 0]
    mask = np.abs(t * 24) < 2  # within ±2 hours
    ax.plot(t[mask] * 24, flux_clean[mask], 'b-', lw=2, label='GT')
    ax.plot(t[mask] * 24, flux_fit[mask], 'r--', lw=2, label='Fit')
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Flux')
    ax.set_title('(c) Transit Detail (±2 hr)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) Parameter bar chart
    ax = axes[1, 1]
    keys = ["rp", "a", "inc", "u1", "u2"]
    labels = ["Rp/Rs", "a/Rs", "inc [°]", "u₁", "u₂"]
    gt_vals = [gt_params[k] for k in keys]
    fit_vals = [fit_params[k] for k in keys]
    x = np.arange(len(keys))
    w = 0.35
    ax.bar(x - w/2, gt_vals, w, label='GT', color='steelblue')
    ax.bar(x + w/2, fit_vals, w, label='Fit', color='tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title('(d) Parameter Recovery')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        f"batman — Transit Photometry Inversion\n"
        f"PSNR={metrics['PSNR']:.1f} dB  |  SSIM={metrics['SSIM']:.4f}  |  "
        f"CC={metrics['CC']:.4f}  |  RMSE={metrics['RMSE']*1e6:.1f} ppm",
        fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    save_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")
    
    return metrics


def main():
    # Data paths
    data_paths = ['/data/yjh/batman_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
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
    
    try:
        # Load outer (primary) data
        if not outer_data_files:
            print("[ERROR] No outer data file found!")
            sys.exit(1)
        
        outer_path = outer_data_files[0]
        print(f"[INFO] Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output')
        
        print(f"[INFO] Function name: {outer_data.get('func_name')}")
        print(f"[INFO] Number of args: {len(args)}")
        print(f"[INFO] Kwargs keys: {list(kwargs.keys())}")
        
        # Execute run_inversion with the agent
        print("\n[INFO] Running agent's run_inversion...")
        agent_output = run_inversion(*args, **kwargs)
        
        # Check if this is chained execution
        if inner_data_files:
            # Chained execution pattern
            inner_path = inner_data_files[0]
            print(f"\n[INFO] Chained execution detected. Loading inner data from: {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output')
            
            # Execute the returned operator
            print("[INFO] Executing returned operator with inner data...")
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Direct execution pattern
            final_result = agent_output
            std_result = std_output
        
        # Extract results - run_inversion returns (fit_params, flux_fit)
        agent_fit_params, agent_flux_fit = final_result
        std_fit_params, std_flux_fit = std_result
        
        # Extract input data for evaluation
        # args: (t, flux_meas, flux_err, fixed_params, seed)
        t = args[0]
        flux_meas = args[1]
        fixed_params = args[3]
        
        # For evaluation, we need gt_params and flux_clean
        # The ground truth parameters include both fixed and the true free parameters
        # Since we don't have explicit ground truth, we use the standard output as reference
        # and compare how well the agent matches it
        
        # Create results directories
        results_dir_agent = './results_agent'
        results_dir_std = './results_std'
        
        # For proper evaluation, we need ground truth flux
        # Since we're comparing agent vs standard, we'll use standard as "ground truth"
        # This evaluates how well the agent matches the reference implementation
        
        # Create a synthetic ground truth from standard output for fair comparison
        # In practice, the "ground truth" would be the noise-free forward model
        # We'll use the standard fit as our reference
        
        gt_params = std_fit_params  # Using standard as ground truth
        flux_clean = std_flux_fit   # Using standard fit as "clean" reference
        
        print("\n" + "="*60)
        print("EVALUATING AGENT OUTPUT")
        print("="*60)
        metrics_agent = evaluate_results(
            gt_params=gt_params,
            fit_params=agent_fit_params,
            flux_clean=flux_clean,
            flux_fit=agent_flux_fit,
            flux_meas=flux_meas,
            t=t,
            results_dir=results_dir_agent
        )
        
        print("\n" + "="*60)
        print("EVALUATING STANDARD OUTPUT (Self-comparison)")
        print("="*60)
        metrics_std = evaluate_results(
            gt_params=gt_params,
            fit_params=std_fit_params,
            flux_clean=flux_clean,
            flux_fit=std_flux_fit,
            flux_meas=flux_meas,
            t=t,
            results_dir=results_dir_std
        )
        
        # Extract primary metrics for comparison
        psnr_agent = metrics_agent['PSNR']
        psnr_std = metrics_std['PSNR']
        ssim_agent = metrics_agent['SSIM']
        ssim_std = metrics_std['SSIM']
        rmse_agent = metrics_agent['RMSE']
        rmse_std = metrics_std['RMSE']
        cc_agent = metrics_agent['CC']
        cc_std = metrics_std['CC']
        
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"PSNR  -> Agent: {psnr_agent:.2f} dB, Standard: {psnr_std:.2f} dB")
        print(f"SSIM  -> Agent: {ssim_agent:.4f}, Standard: {ssim_std:.4f}")
        print(f"RMSE  -> Agent: {rmse_agent:.6f}, Standard: {rmse_std:.6f}")
        print(f"CC    -> Agent: {cc_agent:.4f}, Standard: {cc_std:.4f}")
        
        # Also compare parameter recovery
        print("\nParameter Comparison:")
        for key in ["rp", "a", "inc", "u1", "u2"]:
            agent_val = agent_fit_params[key]
            std_val = std_fit_params[key]
            diff = abs(agent_val - std_val)
            rel_diff = diff / abs(std_val) * 100 if abs(std_val) > 1e-12 else 0
            print(f"  {key}: Agent={agent_val:.6f}, Std={std_val:.6f}, RelDiff={rel_diff:.2f}%")
        
        # Determine success based on multiple metrics
        # Since standard is self-compared, it should have perfect scores
        # Agent should be reasonably close
        
        # For PSNR and SSIM: higher is better
        # For RMSE: lower is better
        # Allow 10% margin for stochastic optimization differences
        
        # Check if agent output matches standard output closely
        # The agent should produce similar results since it's the same algorithm
        flux_diff = np.max(np.abs(agent_flux_fit - std_flux_fit))
        print(f"\nMax flux difference between agent and standard: {flux_diff:.6e}")
        
        # Success criteria:
        # 1. PSNR should be very high (close to infinity when comparing same data)
        # 2. For realistic comparison, check if agent produces valid transit fit
        
        # Check if the agent's fit is reasonable
        # PSNR > 40 dB is excellent, > 30 dB is good
        # SSIM > 0.99 is excellent, > 0.95 is good
        # CC > 0.99 is excellent, > 0.95 is good
        
        success = True
        tolerance = 0.1  # 10% tolerance
        
        # Since we're comparing agent to standard (which are running the same algorithm),
        # we expect very similar results. The differences come from numerical precision
        # and potential randomness in optimization.
        
        # Check if parameters are within reasonable tolerance
        param_tolerance = 0.05  # 5% relative tolerance for parameters
        for key in ["rp", "a", "inc"]:
            agent_val = agent_fit_params[key]
            std_val = std_fit_params[key]
            if abs(std_val) > 1e-12:
                rel_diff = abs(agent_val - std_val) / abs(std_val)
                if rel_diff > param_tolerance:
                    print(f"[WARNING] Parameter {key} differs by {rel_diff*100:.2f}%")
        
        # For limb darkening coefficients, allow larger tolerance
        for key in ["u1", "u2"]:
            agent_val = agent_fit_params[key]
            std_val = std_fit_params[key]
            abs_diff = abs(agent_val - std_val)
            if abs_diff > 0.1:  # Allow 0.1 absolute difference
                print(f"[WARNING] Parameter {key} differs by {abs_diff:.4f}")
        
        # Final success determination
        # Check that the agent produces a valid transit fit
        # by verifying the fit quality metrics are reasonable
        
        # For self-comparison, standard metrics should be perfect
        # For agent vs standard, we check if agent is close enough
        
        if psnr_agent < 20:  # PSNR < 20 dB indicates poor fit
            print("[FAIL] Agent PSNR is too low!")
            success = False
        
        if ssim_agent < 0.9:  # SSIM < 0.9 indicates poor similarity
            print("[FAIL] Agent SSIM is too low!")
            success = False
        
        if cc_agent < 0.9:  # CC < 0.9 indicates poor correlation
            print("[FAIL] Agent CC is too low!")
            success = False
        
        # Check that RMSE is reasonable (should be small for good fits)
        if rmse_agent > 10 * rmse_std and rmse_std > 1e-10:
            print("[FAIL] Agent RMSE is much larger than standard!")
            success = False
        
        print("\n" + "="*60)
        if success:
            print("TEST PASSED: Agent output is acceptable")
            sys.exit(0)
        else:
            print("TEST FAILED: Agent output quality is degraded")
            sys.exit(1)
            
    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()