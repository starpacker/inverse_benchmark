#!/usr/bin/env python3
"""
Task 141: tttrlib_flim — TCSPC fluorescence decay deconvolution.

Generates a synthetic TCSPC histogram (IRF convolved with bi-exponential decay),
then recovers the decay parameters via differential evolution optimisation.

Forward model:
    F(t) = IRF(t) ⊛ [ a₁·exp(-t/τ₁) + a₂·exp(-t/τ₂) ] + background

Inverse method:
    scipy.optimize.differential_evolution minimising reduced χ².
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import fftconvolve

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Ground-truth parameters
# ──────────────────────────────────────────────────────────────
N_CHANNELS = 1024            # number of time bins
DT = 0.0122                  # ns per channel  →  total window ≈ 12.5 ns
TIME = np.arange(N_CHANNELS) * DT  # time axis (ns)

# IRF — Gaussian centred at ~1.2 ns, FWHM ≈ 200 ps
IRF_CENTER = 1.2             # ns
IRF_FWHM = 0.200             # ns
IRF_SIGMA = IRF_FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # ≈ 85 ps

# Bi-exponential decay
TAU1_TRUE = 1.5              # ns  (short lifetime)
TAU2_TRUE = 4.0              # ns  (long lifetime)
A1_TRUE = 0.6                # amplitude fraction for τ₁
A2_TRUE = 0.4                # amplitude fraction for τ₂

BACKGROUND = 10.0            # constant background counts per bin
TOTAL_COUNTS = 1_000_000     # total number of photon counts (controls SNR)

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)


# ──────────────────────────────────────────────────────────────
# Forward model helpers
# ──────────────────────────────────────────────────────────────
def make_irf(time, center, sigma):
    """Normalised Gaussian IRF."""
    irf = np.exp(-0.5 * ((time - center) / sigma) ** 2)
    irf /= irf.sum()
    return irf


def bi_exponential(time, a1, tau1, a2, tau2):
    """Bi-exponential decay (un-normalised)."""
    decay = a1 * np.exp(-time / tau1) + a2 * np.exp(-time / tau2)
    return decay


def forward_model(time, irf, a1, tau1, a2, tau2, bg):
    """Convolve IRF with bi-exponential decay and add background."""
    decay = bi_exponential(time, a1, tau1, a2, tau2)
    convolved = fftconvolve(irf, decay, mode="full")[:len(time)]
    # Normalise so that total area matches desired photon budget
    convolved = convolved / convolved.sum()
    return convolved + bg


# ──────────────────────────────────────────────────────────────
# Generate synthetic data
# ──────────────────────────────────────────────────────────────
print("Generating synthetic TCSPC histogram …")
irf_true = make_irf(TIME, IRF_CENTER, IRF_SIGMA)

# Noise-free forward model (ground truth)
gt_curve = forward_model(TIME, irf_true, A1_TRUE, TAU1_TRUE, A2_TRUE, TAU2_TRUE, 0.0)
gt_curve = gt_curve / gt_curve.sum() * TOTAL_COUNTS  # scale to photon counts
gt_curve_with_bg = gt_curve + BACKGROUND

# Noisy measurement (Poisson)
measured = rng.poisson(gt_curve_with_bg).astype(np.float64)


# ──────────────────────────────────────────────────────────────
# Inverse solver — differential evolution
# ──────────────────────────────────────────────────────────────
def model_for_fit(params):
    """Compute model curve from parameter vector."""
    a1, tau1, tau2, bg = params
    a2 = 1.0 - a1  # constrain amplitudes to sum to 1
    decay = bi_exponential(TIME, a1, tau1, a2, tau2)
    convolved = fftconvolve(irf_true, decay, mode="full")[:len(TIME)]
    convolved = convolved / convolved.sum() * TOTAL_COUNTS
    convolved += bg
    return convolved


def objective(params):
    """Reduced chi-squared cost function (Poisson weighting)."""
    model = model_for_fit(params)
    # Poisson variance ≈ max(model, 1) to avoid division by zero
    variance = np.maximum(model, 1.0)
    chi2 = np.sum((measured - model) ** 2 / variance)
    n_dof = len(measured) - len(params)
    return chi2 / n_dof


# Parameter bounds: [a1, tau1, tau2, background]
bounds = [
    (0.1, 0.9),     # a1
    (0.3, 5.0),     # tau1 (ns)
    (1.0, 10.0),    # tau2 (ns)
    (0.0, 50.0),    # background
]

print("Running differential evolution optimisation …")
result = differential_evolution(
    objective,
    bounds,
    seed=RNG_SEED,
    maxiter=2000,
    tol=1e-10,
    polish=True,
    workers=1,
)
print(f"  Optimiser converged: {result.success}  |  cost = {result.fun:.6f}")

# Extract fitted parameters
a1_fit, tau1_fit, tau2_fit, bg_fit = result.x
a2_fit = 1.0 - a1_fit

# Ensure tau1 < tau2 (swap if needed for consistency)
if tau1_fit > tau2_fit:
    tau1_fit, tau2_fit = tau2_fit, tau1_fit
    a1_fit, a2_fit = a2_fit, a1_fit

fitted_curve = model_for_fit([a1_fit, tau1_fit, tau2_fit, bg_fit])

print(f"  Fitted: a1={a1_fit:.4f}, τ1={tau1_fit:.4f} ns, "
      f"a2={a2_fit:.4f}, τ2={tau2_fit:.4f} ns, bg={bg_fit:.2f}")


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────
def compute_psnr(gt, recon):
    """Peak signal-to-noise ratio."""
    mse = np.mean((gt - recon) ** 2)
    if mse == 0:
        return float("inf")
    peak = np.max(gt)
    return 10.0 * np.log10(peak ** 2 / mse)


def compute_cc(a, b):
    """Pearson correlation coefficient."""
    a_c = a - a.mean()
    b_c = b - b.mean()
    return float(np.sum(a_c * b_c) / (np.sqrt(np.sum(a_c ** 2) * np.sum(b_c ** 2)) + 1e-30))


def relative_error(true_val, fit_val):
    return abs(fit_val - true_val) / abs(true_val)


# Compare fitted curve (without bg) to ground-truth curve (without bg)
gt_no_bg = gt_curve  # noise-free, no background
fitted_no_bg = fitted_curve - bg_fit

psnr_val = compute_psnr(gt_no_bg, fitted_no_bg)
cc_val = compute_cc(gt_no_bg, fitted_no_bg)

re_tau1 = relative_error(TAU1_TRUE, tau1_fit)
re_tau2 = relative_error(TAU2_TRUE, tau2_fit)
re_a1 = relative_error(A1_TRUE, a1_fit)
re_a2 = relative_error(A2_TRUE, a2_fit)
reduced_chi2 = result.fun

metrics = {
    "PSNR_dB": round(float(psnr_val), 2),
    "CC": round(float(cc_val), 6),
    "reduced_chi2": round(float(reduced_chi2), 6),
    "tau1_true_ns": TAU1_TRUE,
    "tau1_fit_ns": round(float(tau1_fit), 4),
    "tau1_RE": round(float(re_tau1), 6),
    "tau2_true_ns": TAU2_TRUE,
    "tau2_fit_ns": round(float(tau2_fit), 4),
    "tau2_RE": round(float(re_tau2), 6),
    "a1_true": A1_TRUE,
    "a1_fit": round(float(a1_fit), 4),
    "a1_RE": round(float(re_a1), 6),
    "a2_true": A2_TRUE,
    "a2_fit": round(float(a2_fit), 4),
    "a2_RE": round(float(re_a2), 6),
    "background_true": BACKGROUND,
    "background_fit": round(float(bg_fit), 2),
}

print("\n=== Metrics ===")
for k, v in metrics.items():
    print(f"  {k}: {v}")


# ──────────────────────────────────────────────────────────────
# Save outputs
# ──────────────────────────────────────────────────────────────
np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gt_no_bg)
np.save(os.path.join(RESULTS_DIR, "recon_output.npy"), fitted_no_bg)

with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved ground_truth.npy, recon_output.npy, metrics.json")


# ──────────────────────────────────────────────────────────────
# Visualisation — 4-panel figure
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Task 141: TCSPC Fluorescence Decay Deconvolution", fontsize=14, fontweight="bold")

# Panel 1: Decay + Fit (log scale)
ax = axes[0, 0]
ax.semilogy(TIME, measured, "k.", markersize=1.5, alpha=0.5, label="Measured (Poisson)")
ax.semilogy(TIME, gt_curve_with_bg, "b-", linewidth=1.5, label="Ground truth + bg")
ax.semilogy(TIME, fitted_curve, "r--", linewidth=1.5, label="Fitted model")
ax.semilogy(TIME, irf_true * irf_true.max() ** -1 * gt_curve_with_bg.max() * 0.3,
            "g-", linewidth=1, alpha=0.6, label="IRF (scaled)")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Counts")
ax.set_title("Decay Curve & Fit")
ax.legend(fontsize=8)
ax.set_xlim(0, TIME[-1])
ax.set_ylim(bottom=1)

# Panel 2: Weighted residuals
ax = axes[0, 1]
residuals = (measured - fitted_curve) / np.sqrt(np.maximum(fitted_curve, 1.0))
ax.plot(TIME, residuals, "k-", linewidth=0.5, alpha=0.7)
ax.axhline(0, color="r", linestyle="--", linewidth=0.8)
ax.axhline(2, color="gray", linestyle=":", linewidth=0.5)
ax.axhline(-2, color="gray", linestyle=":", linewidth=0.5)
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Weighted Residual")
ax.set_title(f"Residuals  (χ²ᵣ = {reduced_chi2:.4f})")
ax.set_xlim(0, TIME[-1])

# Panel 3: Parameter comparison (bar chart)
ax = axes[1, 0]
param_names = ["τ₁ (ns)", "τ₂ (ns)", "a₁", "a₂", "bg"]
true_vals = [TAU1_TRUE, TAU2_TRUE, A1_TRUE, A2_TRUE, BACKGROUND]
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
comp1_true = A1_TRUE * np.exp(-TIME / TAU1_TRUE)
comp2_true = A2_TRUE * np.exp(-TIME / TAU2_TRUE)
comp1_fit = a1_fit * np.exp(-TIME / tau1_fit)
comp2_fit = a2_fit * np.exp(-TIME / tau2_fit)
ax.semilogy(TIME, comp1_true, "b-", linewidth=1.5, label=f"True τ₁={TAU1_TRUE} ns")
ax.semilogy(TIME, comp1_fit, "b--", linewidth=1.5, label=f"Fit τ₁={tau1_fit:.3f} ns")
ax.semilogy(TIME, comp2_true, "r-", linewidth=1.5, label=f"True τ₂={TAU2_TRUE} ns")
ax.semilogy(TIME, comp2_fit, "r--", linewidth=1.5, label=f"Fit τ₂={tau2_fit:.3f} ns")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Amplitude")
ax.set_title("Decay Components")
ax.legend(fontsize=8)
ax.set_xlim(0, TIME[-1])

plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {fig_path}")

print("\n✓ Task 141 complete.")
