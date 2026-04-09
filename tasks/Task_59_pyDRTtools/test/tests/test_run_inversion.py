import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_run_inversion import run_inversion

# Import dependencies for evaluate_results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from scipy.signal import find_peaks
from skimage.metrics import structural_similarity as ssim_fn


def evaluate_results(result_dict, results_dir):
    """
    Compute metrics, save results, and generate visualizations.
    
    Parameters
    ----------
    result_dict : dict
        Dictionary containing:
        - 'freq': frequency array
        - 'tau': relaxation time array
        - 'gamma_gt': ground truth DRT
        - 'gamma_rec': recovered DRT
        - 'Z_clean': clean impedance
        - 'Z_noisy': noisy impedance
        - 'Z_fit': fitted impedance
    results_dir : str
        Directory to save results.
    
    Returns
    -------
    metrics : dict
        Dictionary of computed metrics.
    """
    freq = result_dict['freq']
    tau = result_dict['tau']
    gamma_gt = result_dict['gamma_gt']
    gamma_rec = result_dict['gamma_rec']
    Z_clean = result_dict['Z_clean']
    Z_noisy = result_dict['Z_noisy']
    Z_fit = result_dict['Z_fit']
    
    print("\n[EVAL] Computing metrics ...")
    
    # DRT metrics (normalized)
    g_gt = gamma_gt / max(gamma_gt.max(), 1e-12)
    g_rec = gamma_rec / max(gamma_rec.max(), 1e-12)
    
    cc_drt = float(np.corrcoef(g_gt, g_rec)[0, 1])
    re_drt = float(np.linalg.norm(g_gt - g_rec) / max(np.linalg.norm(g_gt), 1e-12))
    rmse_drt = float(np.sqrt(np.mean((g_gt - g_rec) ** 2)))
    
    data_range = g_gt.max() - g_gt.min()
    mse = np.mean((g_gt - g_rec) ** 2)
    psnr_drt = float(10 * np.log10(data_range ** 2 / max(mse, 1e-30)))
    
    tile_rows = 7
    a2d = np.tile(g_gt, (tile_rows, 1))
    b2d = np.tile(g_rec, (tile_rows, 1))
    ssim_drt = float(ssim_fn(a2d, b2d, data_range=data_range, win_size=7))
    
    # Impedance fit metrics
    Z_resid = Z_clean - Z_fit
    rmse_Z = float(np.sqrt(np.mean(np.abs(Z_resid) ** 2)))
    cc_Z_re = float(np.corrcoef(Z_clean.real, Z_fit.real)[0, 1])
    cc_Z_im = float(np.corrcoef(Z_clean.imag, Z_fit.imag)[0, 1])
    
    # Peak detection
    peaks_gt, _ = find_peaks(g_gt, height=0.1)
    peaks_rec, _ = find_peaks(g_rec, height=0.1)
    
    metrics = {
        "PSNR_DRT": psnr_drt,
        "SSIM_DRT": ssim_drt,
        "CC_DRT": cc_drt,
        "RE_DRT": re_drt,
        "RMSE_DRT": rmse_drt,
        "CC_Z_real": cc_Z_re,
        "CC_Z_imag": cc_Z_im,
        "RMSE_Z": rmse_Z,
        "n_peaks_gt": len(peaks_gt),
        "n_peaks_rec": len(peaks_rec),
    }
    
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")
    
    # Save metrics and data
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(results_dir, "reconstruction.npy"), gamma_rec)
    np.save(os.path.join(results_dir, "ground_truth.npy"), gamma_gt)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (a) DRT
    ax = axes[0, 0]
    ax.semilogx(tau, gamma_gt / max(gamma_gt.max(), 1e-12),
                'b-', lw=2, label='GT')
    ax.semilogx(tau, gamma_rec / max(gamma_rec.max(), 1e-12),
                'r--', lw=2, label='Recovered')
    ax.set_xlabel('τ [s]')
    ax.set_ylabel('γ(τ) [normalised]')
    ax.set_title('(a) Distribution of Relaxation Times')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (b) Nyquist plot
    ax = axes[0, 1]
    ax.plot(Z_clean.real, -Z_clean.imag, 'b-', lw=2, label='GT')
    ax.plot(Z_noisy.real, -Z_noisy.imag, 'k.', ms=3, alpha=0.5, label='Noisy')
    ax.plot(Z_fit.real, -Z_fit.imag, 'r--', lw=2, label='Fit')
    ax.set_xlabel("Z' [Ω]")
    ax.set_ylabel("-Z'' [Ω]")
    ax.set_title('(b) Nyquist Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # (c) Bode magnitude
    ax = axes[1, 0]
    ax.loglog(freq, np.abs(Z_clean), 'b-', lw=2, label='GT')
    ax.loglog(freq, np.abs(Z_noisy), 'k.', ms=3, alpha=0.5, label='Noisy')
    ax.loglog(freq, np.abs(Z_fit), 'r--', lw=2, label='Fit')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('|Z| [Ω]')
    ax.set_title('(c) Bode Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # (d) Bode phase
    ax = axes[1, 1]
    ax.semilogx(freq, np.degrees(np.angle(Z_clean)), 'b-', lw=2, label='GT')
    ax.semilogx(freq, np.degrees(np.angle(Z_noisy)), 'k.', ms=3, alpha=0.5)
    ax.semilogx(freq, np.degrees(np.angle(Z_fit)), 'r--', lw=2, label='Fit')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Phase [°]')
    ax.set_title('(d) Bode Phase')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(
        f"pyDRTtools — DRT Inversion from EIS\n"
        f"PSNR={metrics['PSNR_DRT']:.1f} dB  |  "
        f"SSIM={metrics['SSIM_DRT']:.4f}  |  CC={metrics['CC_DRT']:.4f}",
        fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    save_path = os.path.join(results_dir, "reconstruction_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")
    
    return metrics


def main():
    # Data paths provided
    data_paths = ['/data/yjh/pyDRTtools_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Separate outer (standard) and inner (chained) data files
    outer_files = []
    inner_files = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename or 'parent_' in filename:
            inner_files.append(path)
        else:
            outer_files.append(path)
    
    print(f"[INFO] Outer files: {outer_files}")
    print(f"[INFO] Inner files: {inner_files}")
    
    # Results directory
    results_dir = './test_results'
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # Load outer data
        if not outer_files:
            print("[ERROR] No outer data file found!")
            sys.exit(1)
        
        outer_path = outer_files[0]
        print(f"[INFO] Loading outer data from: {outer_path}")
        
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"[INFO] Outer data keys: {outer_data.keys()}")
        
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        print(f"[INFO] Running run_inversion with args length: {len(args)}, kwargs keys: {kwargs.keys()}")
        
        # Execute the agent's function
        agent_output = run_inversion(*args, **kwargs)
        
        # Check if this is a chained execution pattern
        if inner_files and callable(agent_output):
            # Chained execution
            inner_path = inner_files[0]
            print(f"[INFO] Loading inner data from: {inner_path}")
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            std_result = inner_data.get('output', None)
            
            print(f"[INFO] Running chained function with inner args")
            final_result = agent_output(*inner_args, **inner_kwargs)
        else:
            # Direct execution
            final_result = agent_output
            std_result = std_output
        
        print(f"[INFO] Agent result type: {type(final_result)}")
        print(f"[INFO] Standard result type: {type(std_result)}")
        
        # Check if results are dictionaries with required keys
        if isinstance(final_result, dict) and isinstance(std_result, dict):
            print(f"[INFO] Agent result keys: {final_result.keys()}")
            print(f"[INFO] Standard result keys: {std_result.keys()}")
        
        # Evaluate both results
        print("\n" + "="*60)
        print("EVALUATING AGENT RESULT")
        print("="*60)
        agent_results_dir = os.path.join(results_dir, 'agent')
        metrics_agent = evaluate_results(final_result, agent_results_dir)
        
        print("\n" + "="*60)
        print("EVALUATING STANDARD RESULT")
        print("="*60)
        std_results_dir = os.path.join(results_dir, 'standard')
        metrics_std = evaluate_results(std_result, std_results_dir)
        
        # Extract primary metrics for comparison
        # PSNR and SSIM are "higher is better" metrics
        # RMSE and RE are "lower is better" metrics
        
        psnr_agent = metrics_agent.get('PSNR_DRT', 0)
        psnr_std = metrics_std.get('PSNR_DRT', 0)
        
        ssim_agent = metrics_agent.get('SSIM_DRT', 0)
        ssim_std = metrics_std.get('SSIM_DRT', 0)
        
        cc_agent = metrics_agent.get('CC_DRT', 0)
        cc_std = metrics_std.get('CC_DRT', 0)
        
        rmse_agent = metrics_agent.get('RMSE_DRT', float('inf'))
        rmse_std = metrics_std.get('RMSE_DRT', float('inf'))
        
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"PSNR_DRT  -> Agent: {psnr_agent:.4f}, Standard: {psnr_std:.4f}")
        print(f"SSIM_DRT  -> Agent: {ssim_agent:.4f}, Standard: {ssim_std:.4f}")
        print(f"CC_DRT    -> Agent: {cc_agent:.4f}, Standard: {cc_std:.4f}")
        print(f"RMSE_DRT  -> Agent: {rmse_agent:.4f}, Standard: {rmse_std:.4f}")
        
        # Determine success
        # Allow 10% margin for "higher is better" metrics
        # Allow 10% margin for "lower is better" metrics
        
        margin = 0.10  # 10% margin
        
        success = True
        failure_reasons = []
        
        # Check PSNR (higher is better)
        if psnr_agent < psnr_std * (1 - margin):
            failure_reasons.append(f"PSNR degraded: {psnr_agent:.4f} < {psnr_std * (1 - margin):.4f}")
            success = False
        
        # Check SSIM (higher is better)
        if ssim_agent < ssim_std * (1 - margin):
            failure_reasons.append(f"SSIM degraded: {ssim_agent:.4f} < {ssim_std * (1 - margin):.4f}")
            success = False
        
        # Check CC (higher is better, but can be negative)
        # Use absolute comparison for correlation
        if cc_agent < cc_std - margin:
            failure_reasons.append(f"CC degraded: {cc_agent:.4f} < {cc_std - margin:.4f}")
            success = False
        
        # Check RMSE (lower is better)
        if rmse_agent > rmse_std * (1 + margin):
            failure_reasons.append(f"RMSE increased: {rmse_agent:.4f} > {rmse_std * (1 + margin):.4f}")
            success = False
        
        if success:
            print("\n[SUCCESS] Agent performance is acceptable!")
            sys.exit(0)
        else:
            print("\n[FAILURE] Agent performance degraded significantly!")
            for reason in failure_reasons:
                print(f"  - {reason}")
            sys.exit(1)
            
    except Exception as e:
        print(f"[ERROR] Test failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()