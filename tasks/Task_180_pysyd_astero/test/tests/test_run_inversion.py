import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_run_inversion import run_inversion

# Import dependencies required by evaluate_results
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, binary_dilation
import json

# Inject the referee (evaluation logic) from Reference B

def harvey_comp(freq, zeta, nc):
    """Single Harvey component: P(ν) = ζ / (1 + (ν/ν_c)²)"""
    return zeta / (1.0 + (freq / nc) ** 2)

def bg_model(freq, z1, nc1, z2, nc2, w):
    """Background = 2 Harvey components + white noise."""
    return harvey_comp(freq, z1, nc1) + harvey_comp(freq, z2, nc2) + w

def osc_modes(freq, numax, dnu, sigma_env, height, width):
    """Lorentzian modes modulated by Gaussian envelope."""
    eps = 1.5
    modes = np.zeros_like(freq)
    n_lo = int(np.floor((numax - 4 * sigma_env) / dnu))
    n_hi = int(np.ceil((numax + 4 * sigma_env) / dnu))

    for n in range(max(1, n_lo), n_hi + 1):
        for ell, vis in [(0, 1.0), (1, 0.7), (2, 0.5)]:
            d02 = -0.15 * dnu if ell == 2 else 0.0
            nu_m = dnu * (n + ell / 2.0 + eps) + d02
            if nu_m < freq[0] or nu_m > freq[-1]:
                continue
            env = np.exp(-0.5 * ((nu_m - numax) / sigma_env) ** 2)
            modes += height * env * vis * width ** 2 / ((freq - nu_m) ** 2 + width ** 2)
    return modes

def gauss_env(freq, amp, center, sigma):
    """Gaussian envelope function."""
    return amp * np.exp(-0.5 * ((freq - center) / sigma) ** 2)

def compute_psnr(sig, ref):
    """Compute PSNR in log-space."""
    ls = np.log10(np.maximum(sig, 1e-10))
    lr = np.log10(np.maximum(ref, 1e-10))
    mse = np.mean((ls - lr) ** 2)
    if mse < 1e-30:
        return 100.0
    dr = np.max(lr) - np.min(lr)
    return 10.0 * np.log10(dr ** 2 / mse)

def compute_cc(a, b):
    """Compute cross-correlation."""
    a, b = a - np.mean(a), b - np.mean(b)
    d = np.sqrt(np.sum(a**2) * np.sum(b**2))
    return float(np.sum(a * b) / d) if d > 1e-30 else 0.0

def forward_operator(freq, numax, delta_nu, sigma_env, mode_height, mode_width,
                     harvey_zeta1, harvey_nc1, harvey_zeta2, harvey_nc2, white_noise):
    """
    Forward model: Given oscillation parameters → power spectrum.
    """
    # Background model
    background = bg_model(freq, harvey_zeta1, harvey_nc1, harvey_zeta2, harvey_nc2, white_noise)
    
    # Oscillation modes
    modes = osc_modes(freq, numax, delta_nu, sigma_env, mode_height, mode_width)
    
    # Total power spectrum
    y_pred = background + modes
    
    return y_pred

def evaluate_results(freq, ps_true, ps_obs, inversion_result, params,
                     output_dir='results'):
    """
    Evaluate inversion results and generate visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract results
    numax_est = inversion_result['numax_est']
    dnu_est = inversion_result['dnu_est']
    bg_fit = inversion_result['bg_fit']
    snr = inversion_result['snr']
    lag = inversion_result['lag']
    acf = inversion_result['acf']
    
    # Extract ground truth
    gt_numax = params['gt_numax']
    gt_delta_nu = params['gt_delta_nu']
    sigma_env = params['sigma_env']
    
    # Compute ground truth background
    bg_true = bg_model(freq, params['harvey_zeta1'], params['harvey_nc1'],
                       params['harvey_zeta2'], params['harvey_nc2'], params['white_noise'])
    
    # Compute metrics
    nre = abs(numax_est - gt_numax) / gt_numax
    dre = abs(dnu_est - gt_delta_nu) / gt_delta_nu
    psnr = compute_psnr(bg_fit, bg_true)
    
    m = (freq > 60) & (freq < 200)
    cc = compute_cc(gauss_env(freq[m], 1, numax_est, sigma_env),
                    gauss_env(freq[m], 1, gt_numax, sigma_env))
    
    metrics = {
        "numax_true": gt_numax,
        "numax_estimated": float(round(numax_est, 3)),
        "numax_relative_error": float(round(nre, 6)),
        "delta_nu_true": gt_delta_nu,
        "delta_nu_estimated": float(round(dnu_est, 3)),
        "delta_nu_relative_error": float(round(dre, 6)),
        "background_PSNR_dB": float(round(psnr, 2)),
        "envelope_CC": float(round(cc, 4)),
        "numax_RE_pass": bool(nre < 0.05),
        "delta_nu_RE_pass": bool(dre < 0.05),
    }
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate visualization
    df = freq[1] - freq[0]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Task 180: Asteroseismic Parameter Extraction\n"
        f"νmax = {numax_est:.1f} μHz (GT: {gt_numax}), "
        f"Δν = {dnu_est:.2f} μHz (GT: {gt_delta_nu})",
        fontsize=13, fontweight='bold')

    om = (freq > 60) & (freq < 200)

    ax = axes[0, 0]
    ax.loglog(freq, ps_obs, color='lightgray', lw=0.3, alpha=0.5,
              label='Observed', rasterized=True)
    ax.loglog(freq, ps_true, 'b-', lw=0.8, alpha=0.7, label='True spectrum')
    ax.loglog(freq, bg_true, 'g--', lw=1.5, label='True BG')
    ax.loglog(freq, bg_fit, 'r-', lw=1.5, label='Fitted BG')
    ax.set_xlabel('Frequency (μHz)')
    ax.set_ylabel('Power (ppm²/μHz)')
    ax.set_title('(a) Power Spectrum & Background')
    ax.legend(fontsize=8)
    ax.set_xlim([freq[0], freq[-1]])

    ax = axes[0, 1]
    exc = np.clip(ps_obs / bg_fit - 1.0, -1, 50)
    exc_sm = gaussian_filter1d(exc, int(2.0 / df))
    ax.plot(freq[om], exc[om], color='lightgray', lw=0.3, alpha=0.5, rasterized=True)
    ax.plot(freq[om], exc_sm[om], 'b-', lw=1.0, label='Smoothed')
    ax.axvline(numax_est, color='r', ls='--', lw=1.5, label=f'νmax={numax_est:.1f}')
    ax.axvline(gt_numax, color='g', ls=':', lw=1.5, label=f'GT={gt_numax}')
    ax.set_xlabel('Frequency (μHz)')
    ax.set_ylabel('SNR')
    ax.set_title('(b) Background-Subtracted')
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    et = gauss_env(freq[om], 1, gt_numax, sigma_env)
    ee = gauss_env(freq[om], 1, numax_est, sigma_env)
    ax.plot(freq[om], et, 'g-', lw=2, label='GT')
    ax.plot(freq[om], ee, 'r--', lw=2, label='Est')
    ax.fill_between(freq[om], et, alpha=0.2, color='green')
    ax.fill_between(freq[om], ee, alpha=0.2, color='red')
    ax.set_xlabel('Frequency (μHz)')
    ax.set_ylabel('Normalized Envelope')
    ax.set_title('(c) Envelope: GT vs Estimated')
    ax.legend(fontsize=8)
    ax.text(0.05, 0.9, f'νmax RE={metrics["numax_relative_error"]*100:.2f}%',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax = axes[1, 1]
    show = lag < 30
    sw = max(3, int(0.5 / df))
    asm = gaussian_filter1d(acf[show], sigma=sw)
    ax.plot(lag[show], acf[show], color='lightblue', lw=0.5)
    ax.plot(lag[show], asm, 'b-', lw=1.5, label='Smoothed ACF')
    ax.axvline(dnu_est, color='r', ls='--', lw=2, label=f'Δν={dnu_est:.2f}')
    ax.axvline(gt_delta_nu, color='g', ls=':', lw=2, label=f'GT={gt_delta_nu}')
    ax.set_xlabel('Lag (μHz)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('(d) ACF — Δν')
    ax.legend(fontsize=8)
    ax.text(0.05, 0.9, f'Δν RE={metrics["delta_nu_relative_error"]*100:.2f}%',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig_path = os.path.join(output_dir, 'reconstruction_result.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Figure saved to {fig_path}")
    
    # Save reconstruction
    recon = forward_operator(freq, numax_est, dnu_est, params['sigma_env'],
                             params['mode_height'], params['mode_width'],
                             params['harvey_zeta1'], params['harvey_nc1'],
                             params['harvey_zeta2'], params['harvey_nc2'],
                             params['white_noise'])
    
    np.save(os.path.join(output_dir, 'ground_truth.npy'), ps_true)
    np.save(os.path.join(output_dir, 'reconstruction.npy'), recon)
    
    return metrics


def main():
    """Main test function."""
    # Data paths
    data_paths = ['/data/yjh/pysyd_astero_sandbox_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
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
    
    # Load outer data
    if not outer_files:
        print("[ERROR] No outer data file found!")
        sys.exit(1)
    
    outer_path = outer_files[0]
    print(f"[INFO] Loading outer data from: {outer_path}")
    
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract inputs and expected output
    args = outer_data.get('args', ())
    kwargs = outer_data.get('kwargs', {})
    std_result = outer_data.get('output', None)
    
    print(f"[INFO] Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"[INFO] Number of args: {len(args)}")
    print(f"[INFO] Kwargs keys: {list(kwargs.keys())}")
    
    # Run the agent's implementation
    print("[INFO] Running agent's run_inversion...")
    try:
        agent_result = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"[ERROR] Agent function failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if we have inner data (chained execution)
    if inner_files:
        print("[INFO] Chained execution detected - processing inner data...")
        inner_path = inner_files[0]
        
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        std_result = inner_data.get('output', None)
        
        # Execute the returned callable
        if callable(agent_result):
            print("[INFO] Running inner function...")
            try:
                final_agent_result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"[ERROR] Inner function failed: {e}")
                traceback.print_exc()
                sys.exit(1)
        else:
            final_agent_result = agent_result
    else:
        final_agent_result = agent_result
    
    # Extract freq, ps_obs, params from the original args
    # Based on the function signature: run_inversion(freq, ps_obs, params)
    freq = args[0]
    ps_obs = args[1]
    params = args[2]
    
    print(f"[INFO] freq shape: {freq.shape if hasattr(freq, 'shape') else len(freq)}")
    print(f"[INFO] ps_obs shape: {ps_obs.shape if hasattr(ps_obs, 'shape') else len(ps_obs)}")
    print(f"[INFO] params keys: {list(params.keys()) if isinstance(params, dict) else 'N/A'}")
    
    # We need ps_true for evaluation - it should be derivable from params
    # Generate ps_true using forward_operator
    ps_true = forward_operator(
        freq,
        params['gt_numax'],
        params['gt_delta_nu'],
        params['sigma_env'],
        params['mode_height'],
        params['mode_width'],
        params['harvey_zeta1'],
        params['harvey_nc1'],
        params['harvey_zeta2'],
        params['harvey_nc2'],
        params['white_noise']
    )
    
    # Evaluate agent's result
    print("[INFO] Evaluating agent's result...")
    try:
        agent_metrics = evaluate_results(
            freq, ps_true, ps_obs, final_agent_result, params,
            output_dir='results_agent'
        )
    except Exception as e:
        print(f"[ERROR] Agent evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Evaluate standard result
    print("[INFO] Evaluating standard result...")
    try:
        std_metrics = evaluate_results(
            freq, ps_true, ps_obs, std_result, params,
            output_dir='results_std'
        )
    except Exception as e:
        print(f"[ERROR] Standard evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Print comparison
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nAgent Results:")
    print(f"  numax_estimated: {agent_metrics['numax_estimated']}")
    print(f"  numax_relative_error: {agent_metrics['numax_relative_error']:.6f}")
    print(f"  delta_nu_estimated: {agent_metrics['delta_nu_estimated']}")
    print(f"  delta_nu_relative_error: {agent_metrics['delta_nu_relative_error']:.6f}")
    print(f"  background_PSNR_dB: {agent_metrics['background_PSNR_dB']:.2f}")
    print(f"  envelope_CC: {agent_metrics['envelope_CC']:.4f}")
    print(f"  numax_RE_pass: {agent_metrics['numax_RE_pass']}")
    print(f"  delta_nu_RE_pass: {agent_metrics['delta_nu_RE_pass']}")
    
    print(f"\nStandard Results:")
    print(f"  numax_estimated: {std_metrics['numax_estimated']}")
    print(f"  numax_relative_error: {std_metrics['numax_relative_error']:.6f}")
    print(f"  delta_nu_estimated: {std_metrics['delta_nu_estimated']}")
    print(f"  delta_nu_relative_error: {std_metrics['delta_nu_relative_error']:.6f}")
    print(f"  background_PSNR_dB: {std_metrics['background_PSNR_dB']:.2f}")
    print(f"  envelope_CC: {std_metrics['envelope_CC']:.4f}")
    print(f"  numax_RE_pass: {std_metrics['numax_RE_pass']}")
    print(f"  delta_nu_RE_pass: {std_metrics['delta_nu_RE_pass']}")
    
    # Determine success based on metrics
    # For relative errors, lower is better
    # For PSNR and CC, higher is better
    
    # Compute a combined score (lower relative errors are better, higher PSNR/CC are better)
    # We'll use a simple aggregate: average of relative errors (lower is better)
    agent_error_avg = (agent_metrics['numax_relative_error'] + agent_metrics['delta_nu_relative_error']) / 2
    std_error_avg = (std_metrics['numax_relative_error'] + std_metrics['delta_nu_relative_error']) / 2
    
    print(f"\n[SCORE] Agent average relative error: {agent_error_avg:.6f}")
    print(f"[SCORE] Standard average relative error: {std_error_avg:.6f}")
    
    # Allow some tolerance (10% worse is acceptable)
    tolerance = 1.1
    
    # Check if agent passes the quality thresholds
    agent_passes_numax = agent_metrics['numax_RE_pass']
    agent_passes_dnu = agent_metrics['delta_nu_RE_pass']
    
    # Check if agent is not significantly worse than standard
    agent_not_worse = agent_error_avg <= std_error_avg * tolerance
    
    print(f"\n[CHECK] Agent passes numax threshold (<5%): {agent_passes_numax}")
    print(f"[CHECK] Agent passes delta_nu threshold (<5%): {agent_passes_dnu}")
    print(f"[CHECK] Agent not significantly worse than standard: {agent_not_worse}")
    
    # Final verdict
    if agent_passes_numax and agent_passes_dnu:
        print("\n[SUCCESS] Agent implementation passes all quality thresholds!")
        sys.exit(0)
    elif agent_not_worse:
        print("\n[SUCCESS] Agent implementation is comparable to standard!")
        sys.exit(0)
    else:
        print("\n[FAILURE] Agent implementation does not meet quality requirements!")
        sys.exit(1)


if __name__ == "__main__":
    main()