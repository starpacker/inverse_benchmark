import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d, binary_dilation

import json

import os

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
    
    Args:
        freq: frequency array
        numax: frequency of maximum oscillation power
        delta_nu: large frequency separation
        sigma_env: Gaussian envelope width
        mode_height: peak mode height
        mode_width: Lorentzian HWHM
        harvey_zeta1, harvey_nc1: Harvey component 1 parameters
        harvey_zeta2, harvey_nc2: Harvey component 2 parameters
        white_noise: white noise level
        
    Returns:
        y_pred: predicted power spectrum
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
    
    Args:
        freq: frequency array
        ps_true: true power spectrum
        ps_obs: observed power spectrum
        inversion_result: dictionary from run_inversion
        params: dictionary with ground truth parameters
        output_dir: directory for saving results
        
    Returns:
        metrics: dictionary with evaluation metrics
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
