#!/usr/bin/env python3
"""
Stress Intensity Factor (SIF) Estimation via Williams Series Fitting

Inverse Problem: Given a DIC displacement field around a crack tip,
fit the Williams expansion series to extract Mode I (K_I) and Mode II (K_II)
Stress Intensity Factors.

Williams expansion for displacement near crack tip (polar coords r, θ):
  u_x(r,θ) = Σ_n  A_n * r^(n/2) * f_n^I(θ)  +  B_n * r^(n/2) * f_n^II(θ)
  u_y(r,θ) = Σ_n  A_n * r^(n/2) * g_n^I(θ)  +  B_n * r^(n/2) * g_n^II(θ)

For n=1: K_I = A_1 * sqrt(2π),  K_II = B_1 * sqrt(2π)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ─── Output directory ────────────────────────────────────────────────────────
os.makedirs('results', exist_ok=True)

# ─── Material and geometry parameters ────────────────────────────────────────
E = 210e3       # Young's modulus [MPa]  (210 GPa)
nu = 0.3        # Poisson's ratio
mu = E / (2.0 * (1.0 + nu))          # Shear modulus
kappa = (3.0 - nu) / (1.0 + nu)      # Kolosov constant, plane stress

# Ground-truth SIF values
K_I_true = 30.0     # MPa√m  (Mode I)
K_II_true = 10.0    # MPa√m  (Mode II)

# Number of Williams terms to use (n = 1, 2, ..., N_terms)
N_terms = 5

# ─── Williams displacement basis functions ───────────────────────────────────
# Following the standard Williams expansion for isotropic linear elastic
# material near a crack tip.  Reference: Williams (1957), Westergaard.
#
# For each term index n (n = 1, 2, 3, ...):
#   Mode I displacement:
#     u_x^I = r^(n/2) / (2μ) * [
#       cos(nθ/2) * (κ + n/2 + (-1)^n)
#       - (n/2) * cos((n/2 - 1)θ)                                     ... (*)
#     ]
#   (and similarly for u_y, Mode II)
#
# For simplicity and correctness we implement the FIRST-TERM analytic form
# exactly and higher-order terms via the general Muskhelishvili/Williams
# eigenfunctions.


def williams_mode1_ux(r, theta, n):
    """Mode I contribution to u_x for term index n."""
    rn = r ** (n / 2.0)
    if n % 2 == 1:  # odd n
        val = (kappa + n / 2.0 + ((-1) ** n)) * np.cos(n * theta / 2.0) \
              - (n / 2.0) * np.cos((n / 2.0 - 1.0) * theta)
    else:  # even n
        val = (kappa + n / 2.0 + ((-1) ** n)) * np.cos(n * theta / 2.0) \
              - (n / 2.0) * np.cos((n / 2.0 - 1.0) * theta)
    return rn / (2.0 * mu) * val


def williams_mode1_uy(r, theta, n):
    """Mode I contribution to u_y for term index n."""
    rn = r ** (n / 2.0)
    if n % 2 == 1:
        val = (kappa - n / 2.0 - ((-1) ** n)) * np.sin(n * theta / 2.0) \
              + (n / 2.0) * np.sin((n / 2.0 - 1.0) * theta)
    else:
        val = (kappa - n / 2.0 - ((-1) ** n)) * np.sin(n * theta / 2.0) \
              + (n / 2.0) * np.sin((n / 2.0 - 1.0) * theta)
    return rn / (2.0 * mu) * val


def williams_mode2_ux(r, theta, n):
    """Mode II contribution to u_x for term index n."""
    rn = r ** (n / 2.0)
    val = (kappa + n / 2.0 - ((-1) ** n)) * np.sin(n * theta / 2.0) \
          - (n / 2.0) * np.sin((n / 2.0 - 1.0) * theta)
    return rn / (2.0 * mu) * val


def williams_mode2_uy(r, theta, n):
    """Mode II contribution to u_y for term index n."""
    rn = r ** (n / 2.0)
    val = -(kappa - n / 2.0 + ((-1) ** n)) * np.cos(n * theta / 2.0) \
          + (n / 2.0) * np.cos((n / 2.0 - 1.0) * theta)
    return rn / (2.0 * mu) * val


# ─── Forward model ───────────────────────────────────────────────────────────
def williams_displacement(r, theta, coeffs_I, coeffs_II, n_terms):
    """
    Compute displacement field from Williams coefficients.

    Parameters
    ----------
    r, theta : ndarray  — polar coordinates (r in metres)
    coeffs_I : array of length n_terms — Mode I Williams coefficients A_n
               where K_I = A_1 * sqrt(2π)
    coeffs_II : array of length n_terms — Mode II Williams coefficients B_n
               where K_II = B_1 * sqrt(2π)

    Returns
    -------
    ux, uy : ndarray — displacement components
    """
    ux = np.zeros_like(r)
    uy = np.zeros_like(r)
    for i in range(n_terms):
        n = i + 1
        ux += coeffs_I[i] * williams_mode1_ux(r, theta, n)
        uy += coeffs_I[i] * williams_mode1_uy(r, theta, n)
        ux += coeffs_II[i] * williams_mode2_ux(r, theta, n)
        uy += coeffs_II[i] * williams_mode2_uy(r, theta, n)
    return ux, uy


# ─── Generate synthetic DIC data ────────────────────────────────────────────
print("=" * 60)
print("SIF Estimation via Williams Series Fitting")
print("=" * 60)

# Polar grid around crack tip
Nr, Ntheta = 30, 60
r_vals = np.linspace(1e-3, 10e-3, Nr)      # 1 mm to 10 mm (in metres)
theta_vals = np.linspace(-np.pi * 0.95, np.pi * 0.95, Ntheta)  # avoid ±π singularity
R, THETA = np.meshgrid(r_vals, theta_vals)
r_flat = R.ravel()
theta_flat = THETA.ravel()

# Ground-truth Williams coefficients
# K = A_1 * sqrt(2π)  =>  A_1 = K / sqrt(2π)
coeffs_I_true = np.zeros(N_terms)
coeffs_II_true = np.zeros(N_terms)
coeffs_I_true[0] = K_I_true / np.sqrt(2.0 * np.pi)   # n=1
coeffs_II_true[0] = K_II_true / np.sqrt(2.0 * np.pi)  # n=1

# Add small higher-order terms for realism
np.random.seed(42)
for i in range(1, N_terms):
    coeffs_I_true[i] = np.random.uniform(-0.5, 0.5)
    coeffs_II_true[i] = np.random.uniform(-0.3, 0.3)

print(f"\nGround-truth Williams coefficients (Mode I):  {coeffs_I_true}")
print(f"Ground-truth Williams coefficients (Mode II): {coeffs_II_true}")
print(f"Ground-truth K_I = {K_I_true:.2f} MPa√m")
print(f"Ground-truth K_II = {K_II_true:.2f} MPa√m")

# Generate clean displacement field
ux_clean, uy_clean = williams_displacement(
    r_flat, theta_flat, coeffs_I_true, coeffs_II_true, N_terms
)

# Add Gaussian noise (SNR ~30 dB)
signal_power_ux = np.mean(ux_clean ** 2)
signal_power_uy = np.mean(uy_clean ** 2)
snr_db = 30.0
noise_power_ux = signal_power_ux / (10 ** (snr_db / 10.0))
noise_power_uy = signal_power_uy / (10 ** (snr_db / 10.0))

np.random.seed(123)
noise_ux = np.random.normal(0, np.sqrt(noise_power_ux), ux_clean.shape)
noise_uy = np.random.normal(0, np.sqrt(noise_power_uy), uy_clean.shape)

ux_noisy = ux_clean + noise_ux
uy_noisy = uy_clean + noise_uy

print(f"\nSNR: {snr_db:.0f} dB")
print(f"Number of data points: {len(r_flat)}")

# ─── Build design matrix for linear least-squares ───────────────────────────
# The Williams expansion is LINEAR in the coefficients A_n, B_n.
# So we can assemble:
#   d = M * c
# where d = [ux_data; uy_data], c = [A_1,...,A_N, B_1,...,B_N]
# and M is the design matrix of basis functions.

n_pts = len(r_flat)
n_coeffs = 2 * N_terms  # A_1..A_N + B_1..B_N

# Design matrix: rows = [ux_pt1, ..., ux_ptN, uy_pt1, ..., uy_ptN]
#                cols = [A_1, A_2, ..., A_N, B_1, B_2, ..., B_N]
M = np.zeros((2 * n_pts, n_coeffs))

for i in range(N_terms):
    n = i + 1
    # Mode I columns
    M[:n_pts, i] = williams_mode1_ux(r_flat, theta_flat, n)
    M[n_pts:, i] = williams_mode1_uy(r_flat, theta_flat, n)
    # Mode II columns
    M[:n_pts, N_terms + i] = williams_mode2_ux(r_flat, theta_flat, n)
    M[n_pts:, N_terms + i] = williams_mode2_uy(r_flat, theta_flat, n)

# Data vector
d = np.concatenate([ux_noisy, uy_noisy])

# ─── Inverse solve: Linear least squares ────────────────────────────────────
print("\n--- Solving inverse problem (linear least-squares) ---")

# Ordinary least squares: c = (M^T M)^{-1} M^T d
coeffs_fit, residuals, rank, sv = np.linalg.lstsq(M, d, rcond=None)

A_fit = coeffs_fit[:N_terms]
B_fit = coeffs_fit[N_terms:]

K_I_fit = A_fit[0] * np.sqrt(2.0 * np.pi)
K_II_fit = B_fit[0] * np.sqrt(2.0 * np.pi)

print(f"\nFitted Williams coefficients (Mode I):  {A_fit}")
print(f"Fitted Williams coefficients (Mode II): {B_fit}")
print(f"\nFitted K_I  = {K_I_fit:.4f} MPa√m  (true: {K_I_true:.2f})")
print(f"Fitted K_II = {K_II_fit:.4f} MPa√m  (true: {K_II_true:.2f})")

# ─── Evaluation metrics ─────────────────────────────────────────────────────
K_I_re = abs(K_I_fit - K_I_true) / abs(K_I_true) * 100.0
K_II_re = abs(K_II_fit - K_II_true) / abs(K_II_true) * 100.0

# Reconstructed displacement
ux_fit, uy_fit = williams_displacement(
    r_flat, theta_flat, A_fit, B_fit, N_terms
)

# Displacement RMSE
rmse_ux = np.sqrt(np.mean((ux_fit - ux_clean) ** 2))
rmse_uy = np.sqrt(np.mean((uy_fit - uy_clean) ** 2))
rmse_total = np.sqrt(np.mean((ux_fit - ux_clean) ** 2 + (uy_fit - uy_clean) ** 2))

# R² of the fit (against noisy data)
d_fit = np.concatenate([ux_fit, uy_fit])
ss_res = np.sum((d - d_fit) ** 2)
ss_tot = np.sum((d - np.mean(d)) ** 2)
r_squared = 1.0 - ss_res / ss_tot

print(f"\n{'='*60}")
print(f"EVALUATION METRICS")
print(f"{'='*60}")
print(f"K_I  relative error: {K_I_re:.4f} %")
print(f"K_II relative error: {K_II_re:.4f} %")
print(f"Displacement RMSE (ux): {rmse_ux:.6e}")
print(f"Displacement RMSE (uy): {rmse_uy:.6e}")
print(f"Displacement RMSE (total): {rmse_total:.6e}")
print(f"R² of fit: {r_squared:.6f}")

# ─── Save metrics ───────────────────────────────────────────────────────────
metrics = {
    "K_I_true": float(K_I_true),
    "K_II_true": float(K_II_true),
    "K_I_fitted": float(K_I_fit),
    "K_II_fitted": float(K_II_fit),
    "K_I_relative_error_pct": float(K_I_re),
    "K_II_relative_error_pct": float(K_II_re),
    "displacement_rmse_ux": float(rmse_ux),
    "displacement_rmse_uy": float(rmse_uy),
    "displacement_rmse_total": float(rmse_total),
    "R_squared": float(r_squared),
    "SNR_dB": float(snr_db),
    "N_williams_terms": int(N_terms),
    "n_data_points": int(len(r_flat)),
}
with open('results/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("\nMetrics saved to results/metrics.json")

# ─── Save ground truth and reconstruction ────────────────────────────────────
# Store as structured arrays: columns = [r, theta, ux, uy]
gt_data = np.column_stack([r_flat, theta_flat, ux_clean, uy_clean])
recon_data = np.column_stack([r_flat, theta_flat, ux_fit, uy_fit])
np.save('results/ground_truth.npy', gt_data)
np.save('results/reconstruction.npy', recon_data)
print("Ground truth saved to results/ground_truth.npy")
print("Reconstruction saved to results/reconstruction.npy")

# ─── Visualization ──────────────────────────────────────────────────────────
# Convert polar to Cartesian for plotting
x_cart = r_flat * np.cos(theta_flat) * 1e3  # mm
y_cart = r_flat * np.sin(theta_flat) * 1e3  # mm

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# (a) GT displacement field (u_x)
ax = axes[0, 0]
sc = ax.scatter(x_cart, y_cart, c=ux_clean * 1e6, cmap='RdBu_r', s=15, edgecolors='none')
plt.colorbar(sc, ax=ax, label='u_x [μm]')
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
ax.set_title('(a) Ground Truth u_x')
ax.set_aspect('equal')
ax.plot(0, 0, 'k+', markersize=12, markeredgewidth=2)  # crack tip

# (b) Fitted displacement field (u_x)
ax = axes[0, 1]
sc = ax.scatter(x_cart, y_cart, c=ux_fit * 1e6, cmap='RdBu_r', s=15, edgecolors='none',
                vmin=ux_clean.min() * 1e6, vmax=ux_clean.max() * 1e6)
plt.colorbar(sc, ax=ax, label='u_x [μm]')
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
ax.set_title('(b) Fitted u_x (Williams series)')
ax.set_aspect('equal')
ax.plot(0, 0, 'k+', markersize=12, markeredgewidth=2)

# (c) Displacement error map
ax = axes[1, 0]
error_ux = (ux_fit - ux_clean) * 1e6  # μm
sc = ax.scatter(x_cart, y_cart, c=error_ux, cmap='coolwarm', s=15, edgecolors='none')
plt.colorbar(sc, ax=ax, label='Δu_x [μm]')
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
ax.set_title(f'(c) Error map (RMSE={rmse_total*1e6:.3f} μm)')
ax.set_aspect('equal')
ax.plot(0, 0, 'k+', markersize=12, markeredgewidth=2)

# (d) SIF comparison bar chart
ax = axes[1, 1]
x_bar = np.arange(2)
width = 0.3
gt_vals = [K_I_true, K_II_true]
fit_vals = [K_I_fit, K_II_fit]
bars1 = ax.bar(x_bar - width / 2, gt_vals, width, label='Ground Truth', color='steelblue', alpha=0.85)
bars2 = ax.bar(x_bar + width / 2, fit_vals, width, label='Fitted', color='coral', alpha=0.85)
ax.set_xticks(x_bar)
ax.set_xticklabels(['$K_I$', '$K_{II}$'], fontsize=13)
ax.set_ylabel('SIF [MPa√m]')
ax.set_title(f'(d) SIF Comparison (K_I RE={K_I_re:.2f}%, K_II RE={K_II_re:.2f}%)')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar_group in [bars1, bars2]:
    for bar in bar_group:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

plt.suptitle('SIF Estimation via Williams Series Fitting\n'
             f'(E={E/1e3:.0f} GPa, ν={nu}, plane stress, {N_terms} terms, SNR={snr_db:.0f} dB)',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('results/reconstruction_result.png', dpi=150, bbox_inches='tight')
plt.close()
print("Visualization saved to results/reconstruction_result.png")

print(f"\n{'='*60}")
print("DONE — All outputs saved to results/")
print(f"{'='*60}")
