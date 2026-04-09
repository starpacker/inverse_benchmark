#!/usr/bin/env python
"""
Spectral Peak Fitting / Spectral Deconvolution
===============================================
Inverse Problem: Given a measured spectrum that is a superposition of multiple
peaks (Gaussian, Lorentzian, Voigt), recover the individual peak parameters
(position, width, amplitude, shape).

Uses lmfit (the backend that spectrafit relies on) for robust nonlinear
least-squares fitting of composite peak models.
"""
import matplotlib
matplotlib.use('Agg')

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import voigt_profile

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "repo")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

# ---------------------------------------------------------------------------
# Peak shape functions
# ---------------------------------------------------------------------------

def gaussian(x, amplitude, center, sigma):
    """Gaussian peak."""
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def lorentzian(x, amplitude, center, gamma):
    """Lorentzian peak."""
    return amplitude * gamma**2 / ((x - center)**2 + gamma**2)


def voigt(x, amplitude, center, sigma, gamma):
    """Voigt peak – convolution of Gaussian and Lorentzian."""
    # voigt_profile expects sigma>0, gamma>0
    vp = voigt_profile(x - center, sigma, gamma)
    # Normalize so that peak maximum ≈ amplitude
    vp_max = voigt_profile(0.0, sigma, gamma)
    if vp_max > 0:
        return amplitude * vp / vp_max
    return np.zeros_like(x)


# ---------------------------------------------------------------------------
# Step 1 – Synthesize ground-truth spectrum
# ---------------------------------------------------------------------------
print("[DATA] Generating synthetic multi-peak spectrum ...")

np.random.seed(42)
x = np.linspace(0, 1000, 2000)

# Ground-truth peak parameters:
# (type, amplitude, center, width_param_1, [width_param_2 for Voigt])
true_peaks = [
    {"type": "gaussian",   "amplitude": 8.0,  "center": 200.0, "sigma": 25.0},
    {"type": "lorentzian", "amplitude": 6.0,  "center": 350.0, "gamma": 20.0},
    {"type": "voigt",      "amplitude": 10.0, "center": 500.0, "sigma": 15.0, "gamma": 10.0},
    {"type": "gaussian",   "amplitude": 5.0,  "center": 650.0, "sigma": 30.0},
    {"type": "lorentzian", "amplitude": 7.0,  "center": 800.0, "gamma": 18.0},
]

# Build clean spectrum
clean_spectrum = np.zeros_like(x)
individual_peaks_gt = []
for pk in true_peaks:
    if pk["type"] == "gaussian":
        y = gaussian(x, pk["amplitude"], pk["center"], pk["sigma"])
    elif pk["type"] == "lorentzian":
        y = lorentzian(x, pk["amplitude"], pk["center"], pk["gamma"])
    elif pk["type"] == "voigt":
        y = voigt(x, pk["amplitude"], pk["center"], pk["sigma"], pk["gamma"])
    individual_peaks_gt.append(y)
    clean_spectrum += y

# Linear baseline
baseline_true = 0.5 + 0.001 * x
clean_with_baseline = clean_spectrum + baseline_true

# Add noise (SNR ~ 30)
signal_power = np.mean(clean_spectrum**2)
snr_target = 30.0
noise_power = signal_power / (10 ** (snr_target / 10))
noise = np.random.normal(0, np.sqrt(noise_power), size=x.shape)
measured_spectrum = clean_with_baseline + noise

print(f"[DATA]   x range: [{x.min():.0f}, {x.max():.0f}], {len(x)} points")
print(f"[DATA]   {len(true_peaks)} peaks synthesized")
print(f"[DATA]   Noise std = {np.sqrt(noise_power):.4f}, target SNR = {snr_target} dB")
print("[DATA] Done.")

# ---------------------------------------------------------------------------
# Step 2 – Build lmfit composite model and fit
# ---------------------------------------------------------------------------
print("[RECON] Setting up lmfit composite model ...")

from lmfit import Parameters, minimize, report_fit

def composite_model(params, x, peak_defs):
    """Evaluate the composite model given current parameters."""
    y = params['baseline_intercept'] + params['baseline_slope'] * x
    for i, pk in enumerate(peak_defs):
        amp = params[f'p{i}_amplitude']
        cen = params[f'p{i}_center']
        if pk["type"] == "gaussian":
            sig = params[f'p{i}_sigma']
            y = y + gaussian(x, amp, cen, sig)
        elif pk["type"] == "lorentzian":
            gam = params[f'p{i}_gamma']
            y = y + lorentzian(x, amp, cen, gam)
        elif pk["type"] == "voigt":
            sig = params[f'p{i}_sigma']
            gam = params[f'p{i}_gamma']
            y = y + voigt(x, amp, cen, sig, gam)
    return y


def residual_func(params, x, data, peak_defs):
    """Residual = data - model."""
    return data - composite_model(params, x, peak_defs)


# Initial guesses – perturb from truth by ~10-15 %
params = Parameters()
params.add('baseline_intercept', value=0.3, min=-5, max=5)
params.add('baseline_slope', value=0.0008, min=-0.01, max=0.01)

perturbation_scale = 0.12  # 12 % perturbation
for i, pk in enumerate(true_peaks):
    amp_init = pk["amplitude"] * (1 + perturbation_scale * np.random.randn())
    cen_init = pk["center"] + perturbation_scale * 20 * np.random.randn()
    params.add(f'p{i}_amplitude', value=max(amp_init, 0.1), min=0.01, max=50)
    params.add(f'p{i}_center', value=cen_init, min=pk["center"] - 80, max=pk["center"] + 80)
    if pk["type"] in ("gaussian", "voigt"):
        sig_init = pk["sigma"] * (1 + perturbation_scale * np.random.randn())
        params.add(f'p{i}_sigma', value=max(sig_init, 1.0), min=0.5, max=100)
    if pk["type"] in ("lorentzian", "voigt"):
        gam_init = pk["gamma"] * (1 + perturbation_scale * np.random.randn())
        params.add(f'p{i}_gamma', value=max(gam_init, 1.0), min=0.5, max=100)

print("[RECON] Running least-squares optimization ...")
result = minimize(residual_func, params, args=(x, measured_spectrum, true_peaks),
                  method='leastsq', max_nfev=10000)
print(f"[RECON]   Fit converged: {result.success}")
print(f"[RECON]   Num function evals: {result.nfev}")
print(f"[RECON]   Reduced chi-square: {result.redchi:.6f}")

# Extract fitted spectrum
fitted_spectrum = composite_model(result.params, x, true_peaks)

# Extract individual fitted peaks
fitted_baseline = result.params['baseline_intercept'].value + result.params['baseline_slope'].value * x
individual_peaks_fit = []
for i, pk in enumerate(true_peaks):
    amp = result.params[f'p{i}_amplitude'].value
    cen = result.params[f'p{i}_center'].value
    if pk["type"] == "gaussian":
        sig = result.params[f'p{i}_sigma'].value
        y = gaussian(x, amp, cen, sig)
    elif pk["type"] == "lorentzian":
        gam = result.params[f'p{i}_gamma'].value
        y = lorentzian(x, amp, cen, gam)
    elif pk["type"] == "voigt":
        sig = result.params[f'p{i}_sigma'].value
        gam = result.params[f'p{i}_gamma'].value
        y = voigt(x, amp, cen, sig, gam)
    individual_peaks_fit.append(y)

print("[RECON] Fitting complete.")

# ---------------------------------------------------------------------------
# Step 3 – Evaluation
# ---------------------------------------------------------------------------
print("[EVAL] Computing evaluation metrics ...")

# Parameter-level relative errors
param_errors = {}
for i, pk in enumerate(true_peaks):
    pe = {}
    amp_fit = result.params[f'p{i}_amplitude'].value
    cen_fit = result.params[f'p{i}_center'].value
    pe['amplitude_RE'] = abs(amp_fit - pk['amplitude']) / pk['amplitude']
    pe['center_RE'] = abs(cen_fit - pk['center']) / pk['center']
    if pk["type"] in ("gaussian", "voigt"):
        sig_fit = result.params[f'p{i}_sigma'].value
        pe['sigma_RE'] = abs(sig_fit - pk['sigma']) / pk['sigma']
    if pk["type"] in ("lorentzian", "voigt"):
        gam_fit = result.params[f'p{i}_gamma'].value
        pe['gamma_RE'] = abs(gam_fit - pk['gamma']) / pk['gamma']
    pe['type'] = pk['type']
    pe['true_center'] = pk['center']
    pe['fitted_center'] = float(cen_fit)
    pe['true_amplitude'] = pk['amplitude']
    pe['fitted_amplitude'] = float(amp_fit)
    param_errors[f'peak_{i}'] = pe
    avg_re = np.mean([v for k, v in pe.items() if k.endswith('_RE')])
    print(f"[EVAL]   Peak {i} ({pk['type']}, center={pk['center']}): avg RE = {avg_re:.4f}")

# Curve-level metrics: compare fitted total (without noise) vs clean_with_baseline
# Use clean_with_baseline as the reference "ground truth clean spectrum"
gt_signal = clean_with_baseline

# PSNR
mse = np.mean((gt_signal - fitted_spectrum) ** 2)
data_range = np.max(gt_signal) - np.min(gt_signal)
psnr = 10 * np.log10(data_range**2 / mse) if mse > 0 else float('inf')

# Correlation coefficient
cc = np.corrcoef(gt_signal, fitted_spectrum)[0, 1]

# RMSE
rmse = np.sqrt(mse)

# Residual statistics
residuals = measured_spectrum - fitted_spectrum
res_std = np.std(residuals)

# Mean parameter RE across all peaks
all_re = []
for pk_key, pe in param_errors.items():
    for k, v in pe.items():
        if k.endswith('_RE'):
            all_re.append(v)
mean_param_re = np.mean(all_re)

print(f"[EVAL]   Curve PSNR = {psnr:.2f} dB")
print(f"[EVAL]   Curve CC   = {cc:.6f}")
print(f"[EVAL]   Curve RMSE = {rmse:.6f}")
print(f"[EVAL]   Residual std = {res_std:.6f}")
print(f"[EVAL]   Mean param RE = {mean_param_re:.6f}")
print("[EVAL] Done.")

# ---------------------------------------------------------------------------
# Step 4 – Visualization
# ---------------------------------------------------------------------------
print("[VIS] Creating visualization ...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Spectral Peak Fitting – Inverse Problem", fontsize=16, fontweight='bold')

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

# (a) Measured + Fitted + Individual peaks
ax = axes[0, 0]
ax.plot(x, measured_spectrum, 'k-', alpha=0.4, linewidth=0.5, label='Measured')
ax.plot(x, fitted_spectrum, 'r-', linewidth=2, label='Fitted (total)')
for i, y_fit in enumerate(individual_peaks_fit):
    ax.fill_between(x, fitted_baseline, fitted_baseline + y_fit, alpha=0.3, color=colors[i],
                     label=f'Peak {i} ({true_peaks[i]["type"]})')
ax.plot(x, fitted_baseline, 'k--', linewidth=1, alpha=0.5, label='Baseline')
ax.set_xlabel('Channel / Wavenumber')
ax.set_ylabel('Intensity')
ax.set_title('(a) Measured Spectrum + Fitted Decomposition')
ax.legend(fontsize=7, loc='upper right')
ax.grid(True, alpha=0.3)

# (b) Ground truth vs fitted
ax = axes[0, 1]
ax.plot(x, gt_signal, 'b-', linewidth=2, label='Ground Truth (clean)')
ax.plot(x, fitted_spectrum, 'r--', linewidth=2, label='Fitted')
ax.set_xlabel('Channel / Wavenumber')
ax.set_ylabel('Intensity')
ax.set_title(f'(b) Ground Truth vs Fitted  [PSNR={psnr:.1f} dB, CC={cc:.4f}]')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (c) Residual
ax = axes[1, 0]
ax.plot(x, residuals, 'g-', linewidth=0.5, alpha=0.7)
ax.axhline(0, color='k', linewidth=0.5)
ax.fill_between(x, -2*res_std, 2*res_std, alpha=0.15, color='orange', label=f'±2σ (σ={res_std:.4f})')
ax.set_xlabel('Channel / Wavenumber')
ax.set_ylabel('Residual')
ax.set_title('(c) Residual (Measured − Fitted)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (d) Parameter comparison table
ax = axes[1, 1]
ax.axis('off')
ax.set_title('(d) Parameter Comparison', fontsize=12, fontweight='bold')

# Build table data
col_labels = ['Peak', 'Type', 'Param', 'True', 'Fitted', 'RE (%)']
table_data = []
for i, pk in enumerate(true_peaks):
    pe = param_errors[f'peak_{i}']
    # Amplitude
    table_data.append([f'P{i}', pk['type'][:4], 'Amp',
                       f"{pk['amplitude']:.2f}",
                       f"{pe['fitted_amplitude']:.2f}",
                       f"{pe['amplitude_RE']*100:.2f}"])
    # Center
    table_data.append(['', '', 'Cen',
                       f"{pk['center']:.1f}",
                       f"{pe['fitted_center']:.1f}",
                       f"{pe['center_RE']*100:.2f}"])
    # Width params
    if pk['type'] in ('gaussian', 'voigt'):
        sig_fit = result.params[f'p{i}_sigma'].value
        table_data.append(['', '', 'σ',
                           f"{pk['sigma']:.2f}",
                           f"{sig_fit:.2f}",
                           f"{pe['sigma_RE']*100:.2f}"])
    if pk['type'] in ('lorentzian', 'voigt'):
        gam_fit = result.params[f'p{i}_gamma'].value
        table_data.append(['', '', 'γ',
                           f"{pk['gamma']:.2f}",
                           f"{gam_fit:.2f}",
                           f"{pe['gamma_RE']*100:.2f}"])

table = ax.table(cellText=table_data, colLabels=col_labels,
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.2)
# Color header
for j in range(len(col_labels)):
    table[(0, j)].set_facecolor('#4472C4')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[VIS] Saved figure to {fig_path}")

# ---------------------------------------------------------------------------
# Step 5 – Save outputs
# ---------------------------------------------------------------------------
print("[SAVE] Saving outputs ...")

# Ground truth and reconstruction arrays
gt_path = os.path.join(RESULTS_DIR, "ground_truth.npy")
recon_path = os.path.join(RESULTS_DIR, "reconstruction.npy")
np.save(gt_path, gt_signal)
np.save(recon_path, fitted_spectrum)
print(f"[SAVE]   ground_truth.npy  shape={gt_signal.shape}")
print(f"[SAVE]   reconstruction.npy shape={fitted_spectrum.shape}")

# Metrics JSON
metrics = {
    "task": "spectrafit_peak",
    "inverse_problem": "Spectral peak fitting / spectral deconvolution",
    "method": "lmfit least-squares composite model fitting",
    "num_peaks": len(true_peaks),
    "peak_types": [pk["type"] for pk in true_peaks],
    "psnr_dB": round(float(psnr), 2),
    "correlation_coefficient": round(float(cc), 6),
    "rmse": round(float(rmse), 6),
    "residual_std": round(float(res_std), 6),
    "mean_parameter_relative_error": round(float(mean_param_re), 6),
    "snr_target_dB": snr_target,
    "num_data_points": len(x),
    "fit_converged": bool(result.success),
    "reduced_chi_square": round(float(result.redchi), 6),
    "num_function_evals": int(result.nfev),
    "per_peak_errors": {},
}
for i, pk in enumerate(true_peaks):
    pe = param_errors[f'peak_{i}']
    metrics["per_peak_errors"][f"peak_{i}_{pk['type']}_center{pk['center']}"] = {
        k: round(float(v), 6) if isinstance(v, (float, np.floating)) else v
        for k, v in pe.items()
    }

metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"[SAVE]   metrics.json written")

# Also save x-axis and measured data for reproducibility
np.save(os.path.join(RESULTS_DIR, "x_axis.npy"), x)
np.save(os.path.join(RESULTS_DIR, "measured_spectrum.npy"), measured_spectrum)

print("[SAVE] All outputs saved.")
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"  PSNR  = {psnr:.2f} dB")
print(f"  CC    = {cc:.6f}")
print(f"  RMSE  = {rmse:.6f}")
print(f"  Mean Parameter RE = {mean_param_re:.6f} ({mean_param_re*100:.2f}%)")
print(f"  Fit converged: {result.success}")
print(f"{'='*60}")
