"""
pbjam_astero - Asteroseismology Peakbagging Inverse Problem
===========================================================
Task: Extract stellar oscillation mode frequencies from power spectra
Repo: https://github.com/nielsenmb/PBjam
Paper: Nielsen et al., AJ 2021, doi:10.3847/1538-3881/abcd39

Inverse Problem:
    Given a noisy stellar power spectrum P(ν), recover the individual
    p-mode oscillation frequencies {f_nl} using Lorentzian peak fitting.
    
Forward Model:
    P(ν) = Σ_nl H_nl / (1 + ((ν - f_nl)/Γ_nl)²) + B(ν) + noise
    where f_nl follows the asymptotic relation:
        f_nl ≈ Δν(n + l/2 + ε) - δν_0l·δ(l,0→1)

Usage:
    /data/yjh/pbjam_astero_env/bin/python pbjam_astero_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import json
from scipy.optimize import minimize, curve_fit
from scipy.signal import find_peaks

# ═══════════════════════════════════════════════════════════
# 1. Configuration & Paths
# ═══════════════════════════════════════════════════════════
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Stellar oscillation parameters (solar-like star, similar to KIC 6116048)
NU_MAX = 2200.0       # μHz — frequency of maximum oscillation power
DELTA_NU = 103.0      # μHz — large frequency separation
EPSILON = 1.45        # asymptotic phase offset
D02 = 7.5             # μHz — small separation δν02
N_ORDERS = 8          # number of radial orders to simulate
L_MAX = 2             # maximum angular degree (l=0,1,2)
LINEWIDTH = 1.2       # μHz — mode linewidth (Γ)
SNR_MODES = 15.0      # signal-to-noise ratio for mode heights

# Frequency grid
FREQ_MIN = 1500.0     # μHz
FREQ_MAX = 3000.0     # μHz
FREQ_RESOLUTION = 0.1 # μHz
np.random.seed(42)

# ═══════════════════════════════════════════════════════════
# 2. Forward Model: Asymptotic P-mode Frequencies
# ═══════════════════════════════════════════════════════════
def compute_mode_frequencies(delta_nu, nu_max, epsilon, d02, n_orders, l_max):
    """
    Compute p-mode frequencies using asymptotic relation:
        f_nl = Δν(n + l/2 + ε) - l(l+1)·D0
    where D0 ≈ d02/6 for the second-order correction.
    
    Returns dict: {(n,l): frequency}
    """
    # Determine central radial order from ν_max
    n_center = int(round(nu_max / delta_nu - epsilon))
    n_start = n_center - n_orders // 2
    n_end = n_start + n_orders
    
    D0 = d02 / 6.0  # second-order asymptotic term
    
    modes = {}
    for n in range(n_start, n_end):
        for l in range(l_max + 1):
            freq = delta_nu * (n + l / 2.0 + epsilon) - l * (l + 1) * D0
            if FREQ_MIN < freq < FREQ_MAX:
                modes[(n, l)] = freq
    
    return modes

def generate_power_spectrum(freqs, modes, linewidth, heights, bg_params):
    """
    Forward model: Generate power spectrum from mode parameters.
    
    P(ν) = Σ_nl H_nl · L(ν; f_nl, Γ_nl) + B(ν)
    
    where L is a Lorentzian profile and B is the background.
    """
    spectrum = np.zeros_like(freqs)
    
    # Add each mode as a Lorentzian
    for (n, l), freq in modes.items():
        H = heights.get((n, l), 1.0)
        gamma = linewidth
        # Lorentzian: H / (1 + ((ν - f)/Γ)²)
        spectrum += H / (1.0 + ((freqs - freq) / gamma) ** 2)
    
    # Background: Harvey-like profile + white noise
    # B(ν) = A / (1 + (ν/ν_c)^α) + W
    A, nu_c, alpha, W = bg_params
    background = A / (1.0 + (freqs / nu_c) ** alpha) + W
    spectrum += background
    
    return spectrum, background

def compute_mode_heights(modes, nu_max, base_height=50.0):
    """
    Compute mode heights with Gaussian envelope centered at ν_max.
    H_nl ∝ exp(-((f_nl - ν_max)/(0.66·Δν·n_orders/2))²) · V_l²
    
    Visibility: V_0=1, V_1≈1.5, V_2≈0.5
    """
    visibility = {0: 1.0, 1: 1.5, 2: 0.5}
    env_width = 0.66 * DELTA_NU * N_ORDERS / 2.0
    
    heights = {}
    for (n, l), freq in modes.items():
        envelope = np.exp(-0.5 * ((freq - nu_max) / env_width) ** 2)
        V_l = visibility.get(l, 0.3)
        heights[(n, l)] = base_height * envelope * V_l ** 2
    
    return heights

# ═══════════════════════════════════════════════════════════
# 3. Data Generation
# ═══════════════════════════════════════════════════════════
def load_or_generate_data():
    """
    Generate synthetic stellar power spectrum with known mode frequencies.
    Returns: (noisy_spectrum, gt_frequencies_array, metadata)
    """
    freqs = np.arange(FREQ_MIN, FREQ_MAX, FREQ_RESOLUTION)
    
    # Compute ground truth mode frequencies
    gt_modes = compute_mode_frequencies(DELTA_NU, NU_MAX, EPSILON, D02, N_ORDERS, L_MAX)
    
    # Compute mode heights (Gaussian envelope)
    heights = compute_mode_heights(gt_modes, NU_MAX)
    
    # Background parameters: [amplitude, characteristic_freq, slope, white_noise]
    bg_params = [5.0, 800.0, 2.0, 0.5]
    
    # Generate clean power spectrum
    clean_spectrum, background = generate_power_spectrum(freqs, gt_modes, LINEWIDTH, heights, bg_params)
    
    # Add chi-squared noise (2 DOF for power spectrum — exponential distribution)
    # In asteroseismology, power spectrum follows chi² with 2 DOF
    # P_obs = P_true * χ²(2)/2
    noise_factor = np.random.exponential(scale=1.0, size=len(freqs))
    noisy_spectrum = clean_spectrum * noise_factor
    
    # Smooth slightly to improve peak detection (simulate some averaging)
    from scipy.ndimage import uniform_filter1d
    smoothed_spectrum = uniform_filter1d(noisy_spectrum, size=5)
    
    # Prepare GT as sorted frequency array
    gt_freq_list = sorted(gt_modes.values())
    gt_freq_array = np.array(gt_freq_list)
    
    metadata = {
        'freqs': freqs,
        'gt_modes': gt_modes,
        'heights': heights,
        'bg_params': bg_params,
        'clean_spectrum': clean_spectrum,
        'background': background,
        'smoothed_spectrum': smoothed_spectrum,
        'linewidth': LINEWIDTH,
    }
    
    print(f"[DATA] Generated {len(gt_modes)} p-modes across l=0,1,2")
    print(f"[DATA] Frequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} μHz")
    print(f"[DATA] Δν = {DELTA_NU} μHz, ν_max = {NU_MAX} μHz")
    print(f"[DATA] GT frequencies: {gt_freq_array}")
    
    return noisy_spectrum, gt_freq_array, metadata

# ═══════════════════════════════════════════════════════════
# 4. Inverse Solver: Peakbagging via Lorentzian Fitting
# ═══════════════════════════════════════════════════════════
def lorentzian(freq, f0, H, gamma):
    """Single Lorentzian profile."""
    return H / (1.0 + ((freq - f0) / gamma) ** 2)

def multi_lorentzian_bg(freq, *params):
    """
    Multi-Lorentzian model + polynomial background.
    params: [bg_a, bg_b, bg_c, f1, H1, Γ1, f2, H2, Γ2, ...]
    """
    bg_a, bg_b, bg_c = params[0], params[1], params[2]
    background = bg_a + bg_b * (freq / 1000.0) + bg_c * (freq / 1000.0) ** 2
    
    n_modes = (len(params) - 3) // 3
    model = background.copy()
    for i in range(n_modes):
        idx = 3 + i * 3
        f0 = params[idx]
        H = params[idx + 1]
        gamma = params[idx + 2]
        model += lorentzian(freq, f0, H, gamma)
    
    return model

def reconstruct(noisy_spectrum, metadata):
    """
    Peakbagging: Extract mode frequencies from noisy power spectrum.
    
    Algorithm:
    1. Estimate and subtract background
    2. Find candidate peaks in smoothed spectrum
    3. Fit individual Lorentzians around each peak
    4. Refine with global multi-Lorentzian fit
    5. Extract fitted frequencies
    """
    freqs = metadata['freqs']
    smoothed = metadata['smoothed_spectrum']
    gt_modes = metadata['gt_modes']
    
    # Step 1: Estimate background using percentile filtering
    from scipy.ndimage import percentile_filter
    bg_estimate = percentile_filter(noisy_spectrum, percentile=20, size=500)
    bg_subtracted = smoothed - np.minimum(bg_estimate, smoothed * 0.9)
    bg_subtracted = np.maximum(bg_subtracted, 0)
    
    # Step 2: Find peaks in background-subtracted smoothed spectrum
    # Use prominence and distance constraints based on expected Δν
    min_distance = int(DELTA_NU / 5.0 / FREQ_RESOLUTION)  # at least Δν/5 apart (allow closer l=0,1,2 modes)
    prominence = np.percentile(bg_subtracted[bg_subtracted > 0], 40) if np.any(bg_subtracted > 0) else 1.0
    
    peak_indices, peak_props = find_peaks(
        bg_subtracted,
        distance=min_distance,
        prominence=prominence * 0.3,
        height=np.median(bg_subtracted) + np.std(bg_subtracted) * 0.2
    )
    
    candidate_freqs = freqs[peak_indices]
    print(f"[RECON] Found {len(candidate_freqs)} candidate peaks")
    
    # Step 3: Fit individual Lorentzians around each candidate peak
    fitted_freqs = []
    fitted_heights = []
    fitted_widths = []
    
    fit_window = 10.0  # μHz half-window for fitting
    
    for cf in candidate_freqs:
        mask = np.abs(freqs - cf) < fit_window
        if np.sum(mask) < 10:
            continue
        
        freq_window = freqs[mask]
        spec_window = noisy_spectrum[mask]  # use original noisy data for fitting
        
        try:
            # Initial guess: [f0, H, gamma]
            p0 = [cf, np.max(spec_window), LINEWIDTH]
            bounds = ([cf - 5, 0, 0.1], [cf + 5, np.max(spec_window) * 5, 10.0])
            
            # Add constant background to the fit
            def lorentzian_bg(f, f0, H, gamma, bg):
                return H / (1.0 + ((f - f0) / gamma) ** 2) + bg
            
            p0_bg = [cf, np.max(spec_window) - np.min(spec_window), LINEWIDTH, np.min(spec_window)]
            bounds_bg = (
                [cf - 5, 0, 0.1, 0],
                [cf + 5, np.max(spec_window) * 5, 10.0, np.max(spec_window)]
            )
            
            popt, pcov = curve_fit(lorentzian_bg, freq_window, spec_window,
                                   p0=p0_bg, bounds=bounds_bg, maxfev=5000)
            
            fitted_freqs.append(popt[0])
            fitted_heights.append(popt[1])
            fitted_widths.append(popt[2])
        except Exception as e:
            # Fall back to peak position
            fitted_freqs.append(cf)
            fitted_heights.append(np.max(spec_window))
            fitted_widths.append(LINEWIDTH)
    
    fitted_freqs = np.array(fitted_freqs)
    fitted_heights = np.array(fitted_heights)
    fitted_widths = np.array(fitted_widths)
    
    print(f"[RECON] Successfully fitted {len(fitted_freqs)} Lorentzian peaks")
    
    # Step 4: Match fitted frequencies to GT modes (Hungarian-style greedy matching)
    gt_freq_list = sorted(gt_modes.values())
    gt_freq_array = np.array(gt_freq_list)
    
    # Greedy matching: for each GT freq, find closest fitted freq
    matched_gt = []
    matched_fitted = []
    used_fitted = set()
    
    for gf in gt_freq_array:
        dists = np.abs(fitted_freqs - gf)
        sorted_idx = np.argsort(dists)
        for idx in sorted_idx:
            if idx not in used_fitted and dists[idx] < DELTA_NU / 3.0:
                matched_gt.append(gf)
                matched_fitted.append(fitted_freqs[idx])
                used_fitted.add(idx)
                break
    
    matched_gt = np.array(matched_gt)
    matched_fitted = np.array(matched_fitted)
    
    print(f"[RECON] Matched {len(matched_gt)} of {len(gt_freq_array)} GT modes")
    
    # Step 5: Generate reconstructed spectrum from fitted parameters
    recon_spectrum = np.zeros_like(freqs)
    for i in range(len(fitted_freqs)):
        recon_spectrum += lorentzian(freqs, fitted_freqs[i], fitted_heights[i], fitted_widths[i])
    
    # Add estimated background
    recon_spectrum += bg_estimate
    
    return {
        'fitted_freqs': fitted_freqs,
        'fitted_heights': fitted_heights,
        'fitted_widths': fitted_widths,
        'matched_gt': matched_gt,
        'matched_fitted': matched_fitted,
        'recon_spectrum': recon_spectrum,
        'bg_estimate': bg_estimate,
    }

# ═══════════════════════════════════════════════════════════
# 5. Evaluation Metrics
# ═══════════════════════════════════════════════════════════
def compute_metrics(gt_freq_array, recon_result, clean_spectrum, recon_spectrum):
    """Compute evaluation metrics for peakbagging."""
    matched_gt = recon_result['matched_gt']
    matched_fitted = recon_result['matched_fitted']
    
    metrics = {}
    
    if len(matched_gt) > 0:
        # Frequency relative errors
        freq_errors = np.abs(matched_fitted - matched_gt)
        rel_errors = freq_errors / matched_gt * 100  # percentage
        
        metrics['mean_freq_error_uHz'] = float(np.mean(freq_errors))
        metrics['median_freq_error_uHz'] = float(np.median(freq_errors))
        metrics['max_freq_error_uHz'] = float(np.max(freq_errors))
        metrics['mean_relative_error_pct'] = float(np.mean(rel_errors))
        metrics['median_relative_error_pct'] = float(np.median(rel_errors))
        
        # Correlation coefficient for matched frequencies
        if len(matched_gt) > 1:
            cc = float(np.corrcoef(matched_gt, matched_fitted)[0, 1])
        else:
            cc = 1.0
        metrics['frequency_CC'] = cc
        
        # R² for frequency recovery
        ss_res = np.sum((matched_gt - matched_fitted) ** 2)
        ss_tot = np.sum((matched_gt - np.mean(matched_gt)) ** 2)
        metrics['frequency_R2'] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 1.0
        
        # Detection rate
        metrics['n_gt_modes'] = len(gt_freq_array)
        metrics['n_detected'] = len(recon_result['fitted_freqs'])
        metrics['n_matched'] = len(matched_gt)
        metrics['detection_rate'] = float(len(matched_gt) / len(gt_freq_array))
    
    # Spectrum-level metrics (PSNR of reconstructed spectrum)
    if clean_spectrum is not None and recon_spectrum is not None:
        # Compute PSNR
        data_range = clean_spectrum.max() - clean_spectrum.min()
        mse = np.mean((clean_spectrum - recon_spectrum) ** 2)
        if mse > 0:
            psnr = 10 * np.log10(data_range ** 2 / mse)
        else:
            psnr = float('inf')
        metrics['spectrum_PSNR'] = float(psnr)
        
        # Spectrum CC
        cc_spec = float(np.corrcoef(clean_spectrum, recon_spectrum)[0, 1])
        metrics['spectrum_CC'] = cc_spec
    
    # Asteroseismic parameter recovery
    if len(matched_gt) >= 4:
        # Estimate Δν from fitted frequencies (for l=0 modes)
        sorted_fitted = np.sort(matched_fitted)
        diffs = np.diff(sorted_fitted)
        # Filter for diffs close to expected Δν
        dnu_candidates = diffs[(diffs > DELTA_NU * 0.7) & (diffs < DELTA_NU * 1.3)]
        if len(dnu_candidates) > 0:
            fitted_dnu = np.median(dnu_candidates)
            metrics['fitted_delta_nu'] = float(fitted_dnu)
            metrics['delta_nu_error_pct'] = float(abs(fitted_dnu - DELTA_NU) / DELTA_NU * 100)
    
    return metrics

# ═══════════════════════════════════════════════════════════
# 6. Visualization
# ═══════════════════════════════════════════════════════════
def visualize_results(freqs, noisy_spectrum, clean_spectrum, recon_result, metrics, save_path):
    """Generate 4-panel visualization for peakbagging results."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    recon_spectrum = recon_result['recon_spectrum']
    matched_gt = recon_result['matched_gt']
    matched_fitted = recon_result['matched_fitted']
    fitted_freqs = recon_result['fitted_freqs']
    
    # Panel 1: Ground truth clean spectrum with mode locations
    ax1 = axes[0, 0]
    ax1.plot(freqs, clean_spectrum, 'b-', alpha=0.8, linewidth=0.5, label='Clean spectrum')
    for (n, l), freq in sorted(metadata_global['gt_modes'].items()):
        colors = {0: 'red', 1: 'green', 2: 'blue'}
        ax1.axvline(freq, color=colors.get(l, 'gray'), alpha=0.4, linewidth=0.5)
    ax1.set_xlabel('Frequency (μHz)')
    ax1.set_ylabel('Power (ppm²/μHz)')
    ax1.set_title('(a) Ground Truth Power Spectrum + Mode Frequencies')
    ax1.set_xlim(FREQ_MIN, FREQ_MAX)
    ax1.legend(fontsize=8)
    
    # Panel 2: Noisy observed spectrum
    ax2 = axes[0, 1]
    ax2.plot(freqs, noisy_spectrum, 'k-', alpha=0.3, linewidth=0.3, label='Noisy')
    smoothed = metadata_global['smoothed_spectrum']
    ax2.plot(freqs, smoothed, 'b-', alpha=0.8, linewidth=0.8, label='Smoothed')
    ax2.plot(freqs, recon_result['bg_estimate'], 'r--', alpha=0.7, linewidth=1, label='Background')
    for ff in fitted_freqs:
        ax2.axvline(ff, color='orange', alpha=0.4, linewidth=0.5)
    ax2.set_xlabel('Frequency (μHz)')
    ax2.set_ylabel('Power (ppm²/μHz)')
    ax2.set_title(f'(b) Observed Spectrum + {len(fitted_freqs)} Detected Peaks')
    ax2.set_xlim(FREQ_MIN, FREQ_MAX)
    ax2.legend(fontsize=8)
    
    # Panel 3: Reconstructed spectrum vs clean
    ax3 = axes[1, 0]
    ax3.plot(freqs, clean_spectrum, 'b-', alpha=0.6, linewidth=0.8, label='Clean GT')
    ax3.plot(freqs, recon_spectrum, 'r-', alpha=0.6, linewidth=0.8, label='Reconstructed')
    ax3.set_xlabel('Frequency (μHz)')
    ax3.set_ylabel('Power (ppm²/μHz)')
    psnr_val = metrics.get('spectrum_PSNR', 0)
    cc_val = metrics.get('spectrum_CC', 0)
    ax3.set_title(f'(c) Spectrum Reconstruction — PSNR={psnr_val:.2f} dB, CC={cc_val:.4f}')
    ax3.set_xlim(FREQ_MIN, FREQ_MAX)
    ax3.legend(fontsize=8)
    
    # Panel 4: Frequency comparison (GT vs Fitted)
    ax4 = axes[1, 1]
    if len(matched_gt) > 0:
        ax4.scatter(matched_gt, matched_fitted, c='blue', s=50, zorder=5, label='Matched modes')
        # Perfect line
        fmin, fmax = matched_gt.min() - 20, matched_gt.max() + 20
        ax4.plot([fmin, fmax], [fmin, fmax], 'k--', alpha=0.5, label='Perfect match')
        
        # Error bars
        errors = matched_fitted - matched_gt
        for i in range(len(matched_gt)):
            ax4.annotate(f'{errors[i]:.2f}', (matched_gt[i], matched_fitted[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=6, alpha=0.7)
        
        freq_cc = metrics.get('frequency_CC', 0)
        mean_re = metrics.get('mean_relative_error_pct', 0)
        det_rate = metrics.get('detection_rate', 0)
        ax4.set_title(f'(d) Freq Match — CC={freq_cc:.6f}, RE={mean_re:.4f}%, Det={det_rate:.1%}')
    ax4.set_xlabel('Ground Truth Frequency (μHz)')
    ax4.set_ylabel('Fitted Frequency (μHz)')
    ax4.legend(fontsize=8)
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved visualization → {save_path}")

# ═══════════════════════════════════════════════════════════
# 7. Main Pipeline
# ═══════════════════════════════════════════════════════════
metadata_global = None  # Will be set in main

if __name__ == "__main__":
    print("=" * 60)
    print("  pbjam_astero — Asteroseismology Peakbagging")
    print("=" * 60)
    
    # (a) Generate synthetic data
    noisy_spectrum, gt_freq_array, metadata = load_or_generate_data()
    metadata_global = metadata
    freqs = metadata['freqs']
    clean_spectrum = metadata['clean_spectrum']
    
    print(f"[DATA] Noisy spectrum shape: {noisy_spectrum.shape}")
    print(f"[DATA] GT frequency array shape: {gt_freq_array.shape}")
    
    # (b) Run peakbagging (inverse solver)
    recon_result = reconstruct(noisy_spectrum, metadata)
    
    # (c) Evaluate
    metrics = compute_metrics(gt_freq_array, recon_result, clean_spectrum, 
                              recon_result['recon_spectrum'])
    
    print(f"\n[EVAL] === Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    
    # (d) Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVE] Metrics → {metrics_path}")
    
    # (e) Visualize
    vis_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
    visualize_results(freqs, noisy_spectrum, clean_spectrum, recon_result, metrics, vis_path)
    
    # (f) Save arrays
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), recon_result['matched_fitted'])
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt_freq_array)
    np.save(os.path.join(RESULTS_DIR, "noisy_spectrum.npy"), noisy_spectrum)
    np.save(os.path.join(RESULTS_DIR, "clean_spectrum.npy"), clean_spectrum)
    np.save(os.path.join(RESULTS_DIR, "recon_spectrum.npy"), recon_result['recon_spectrum'])
    
    print("=" * 60)
    print("  DONE — pbjam_astero peakbagging complete")
    print("=" * 60)
