"""
Task 122: Electrochemical Impedance Spectroscopy (EIS) Fitting
Inverse Problem: Fit equivalent circuit model (Randles circuit) to measured
complex impedance spectra Z(ω) to recover circuit parameters.

Circuit: R0-p(R1,C1)-W  (Randles circuit)
  R0 = series/ohmic resistance
  R1 = charge transfer resistance
  C1 = double-layer capacitance
  σ_W = Warburg coefficient

Z(ω) = R0 + Z_RC(ω) + Z_W(ω)
where Z_RC = R1 / (1 + jωR1C1)   (parallel RC)
      Z_W  = σ_W / sqrt(ω) * (1 - j)  (Warburg impedance)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import json
import os

# ──────────────────────────────────────────────────────────
# 1. Ground truth parameters
# ──────────────────────────────────────────────────────────
GT_PARAMS = {
    'R0': 50.0,       # Ohm  – series resistance
    'R1': 100.0,      # Ohm  – charge transfer resistance
    'C1': 1e-6,       # F    – double-layer capacitance
    'sigma_W': 50.0,  # Ohm·s^(-1/2) – Warburg coefficient
}

PARAM_ORDER = ['R0', 'R1', 'C1', 'sigma_W']

# ──────────────────────────────────────────────────────────
# 2. Forward model
# ──────────────────────────────────────────────────────────
def randles_impedance(freq, R0, R1, C1, sigma_W):
    """Compute complex impedance of a Randles circuit with Warburg element.

    Parameters
    ----------
    freq : ndarray
        Frequency array in Hz.
    R0, R1, C1, sigma_W : float
        Circuit parameters.

    Returns
    -------
    Z : ndarray (complex)
        Complex impedance at each frequency.
    """
    omega = 2.0 * np.pi * freq
    # Parallel RC element: Z_RC = R1 / (1 + j*omega*R1*C1)
    Z_RC = R1 / (1.0 + 1j * omega * R1 * C1)
    # Warburg element: Z_W = sigma_W / sqrt(omega) * (1 - j)
    Z_W = sigma_W / np.sqrt(omega) * (1.0 - 1j)
    # Total impedance
    Z = R0 + Z_RC + Z_W
    return Z


def forward(params_vec, freq):
    """Wrapper: parameter vector -> complex impedance."""
    R0, R1, C1, sigma_W = params_vec
    return randles_impedance(freq, R0, R1, C1, sigma_W)

# ──────────────────────────────────────────────────────────
# 3. Frequency grid & ground truth spectrum
# ──────────────────────────────────────────────────────────
freq = np.logspace(-2, 6, 60)  # 0.01 Hz to 1 MHz, 60 points

gt_vec = np.array([GT_PARAMS[k] for k in PARAM_ORDER])
Z_true = forward(gt_vec, freq)

# ──────────────────────────────────────────────────────────
# 4. Add noise (1.5% of |Z| Gaussian noise to Re and Im)
# ──────────────────────────────────────────────────────────
np.random.seed(42)
noise_level = 0.015
noise_re = noise_level * np.abs(Z_true) * np.random.randn(len(freq))
noise_im = noise_level * np.abs(Z_true) * np.random.randn(len(freq))
Z_noisy = Z_true + noise_re + 1j * noise_im

# ──────────────────────────────────────────────────────────
# 5. Inverse solver – nonlinear least squares via L-BFGS-B
# ──────────────────────────────────────────────────────────
def objective(params_vec, freq, Z_meas):
    """Weighted least-squares objective on real + imaginary parts."""
    Z_model = forward(params_vec, freq)
    # Weight by 1/|Z_meas| so all frequencies contribute equally
    w = 1.0 / np.abs(Z_meas)
    residual_re = (Z_model.real - Z_meas.real) * w
    residual_im = (Z_model.imag - Z_meas.imag) * w
    return 0.5 * np.sum(residual_re**2 + residual_im**2)


# Initial guesses – try multiple starting points to avoid local minima
bounds = [(1.0, 500.0),    # R0
          (1.0, 1000.0),   # R1
          (1e-9, 1e-3),    # C1
          (1.0, 500.0)]    # sigma_W

initial_guesses = [
    np.array([30.0, 150.0, 5e-6, 80.0]),
    np.array([60.0, 80.0, 2e-6, 30.0]),
    np.array([40.0, 120.0, 8e-7, 60.0]),
    np.array([70.0, 50.0, 3e-6, 40.0]),
    np.array([20.0, 200.0, 1e-5, 20.0]),
]

best_result = None
best_cost = np.inf

for x0 in initial_guesses:
    res = minimize(objective, x0, args=(freq, Z_noisy),
                   method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-14})
    if res.fun < best_cost:
        best_cost = res.fun
        best_result = res

# Refine with Nelder-Mead (derivative-free, good for noisy landscapes)
from scipy.optimize import differential_evolution

de_result = differential_evolution(objective, bounds, args=(freq, Z_noisy),
                                   seed=42, maxiter=2000, tol=1e-14,
                                   polish=True, workers=1)
if de_result.fun < best_cost:
    best_cost = de_result.fun
    best_result = de_result

result = best_result
fitted_params = result.x
Z_fitted = forward(fitted_params, freq)

print("Optimization converged:", result.success)
print(f"Objective value: {result.fun:.6e}")

# ──────────────────────────────────────────────────────────
# 6. Metrics
# ──────────────────────────────────────────────────────────
# Parameter relative errors
param_errors = {}
for i, name in enumerate(PARAM_ORDER):
    gt_val = gt_vec[i]
    fit_val = fitted_params[i]
    rel_err = np.abs(fit_val - gt_val) / np.abs(gt_val)
    param_errors[name] = {
        'ground_truth': float(gt_val),
        'fitted': float(fit_val),
        'relative_error': float(rel_err),
    }
    print(f"  {name}: GT={gt_val:.4e}, Fit={fit_val:.4e}, RelErr={rel_err:.4f}")

# Spectral RMSE (on complex impedance)
rmse = np.sqrt(np.mean(np.abs(Z_fitted - Z_true)**2))

# Spectral PSNR
max_Z = np.max(np.abs(Z_true))
psnr = 20.0 * np.log10(max_Z / rmse) if rmse > 0 else float('inf')

# Correlation coefficient between fitted and true spectra
# Treat as 1D real signal: concatenate real and imag parts
sig_true = np.concatenate([Z_true.real, Z_true.imag])
sig_fit = np.concatenate([Z_fitted.real, Z_fitted.imag])
cc = float(np.corrcoef(sig_true, sig_fit)[0, 1])

# Mean parameter relative error
mean_re = float(np.mean([v['relative_error'] for v in param_errors.values()]))

print(f"\nSpectral RMSE: {rmse:.4f} Ohm")
print(f"Spectral PSNR: {psnr:.2f} dB")
print(f"Correlation Coefficient: {cc:.6f}")
print(f"Mean Parameter Relative Error: {mean_re:.4f}")

# ──────────────────────────────────────────────────────────
# 7. Save results
# ──────────────────────────────────────────────────────────
os.makedirs('results', exist_ok=True)

metrics = {
    'task': 'impedance_eis',
    'task_number': 122,
    'description': 'Electrochemical Impedance Spectroscopy (EIS) Randles circuit fitting',
    'psnr_db': round(float(psnr), 2),
    'rmse_ohm': round(float(rmse), 4),
    'correlation_coefficient': round(cc, 6),
    'mean_parameter_relative_error': round(mean_re, 6),
    'parameters': param_errors,
    'noise_level_percent': noise_level * 100,
    'num_frequencies': len(freq),
    'optimizer': 'L-BFGS-B',
    'converged': bool(result.success),
}

with open('results/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Save spectra as npy (stacked real+imag, shape 2×N)
np.save('results/ground_truth.npy', np.stack([Z_true.real, Z_true.imag]))
np.save('results/reconstruction.npy', np.stack([Z_fitted.real, Z_fitted.imag]))

# ──────────────────────────────────────────────────────────
# 8. Visualization
# ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Task 122: EIS Randles Circuit Fitting', fontsize=15, fontweight='bold')

# --- (a) Nyquist plot ---
ax = axes[0, 0]
ax.plot(Z_true.real, -Z_true.imag, 'b-', linewidth=2, label='Ground Truth')
ax.plot(Z_noisy.real, -Z_noisy.imag, 'k.', markersize=5, alpha=0.5, label='Noisy Data')
ax.plot(Z_fitted.real, -Z_fitted.imag, 'r--', linewidth=2, label='Fitted')
ax.set_xlabel('Z_real (Ω)', fontsize=11)
ax.set_ylabel('-Z_imag (Ω)', fontsize=11)
ax.set_title('(a) Nyquist Plot', fontsize=12)
ax.legend(fontsize=9)
ax.set_aspect('equal', adjustable='datalim')
ax.grid(True, alpha=0.3)

# --- (b) Bode plot: |Z| and phase vs frequency ---
ax = axes[0, 1]
ax.loglog(freq, np.abs(Z_true), 'b-', linewidth=2, label='|Z| GT')
ax.loglog(freq, np.abs(Z_noisy), 'k.', markersize=4, alpha=0.4, label='|Z| Noisy')
ax.loglog(freq, np.abs(Z_fitted), 'r--', linewidth=2, label='|Z| Fitted')
ax.set_xlabel('Frequency (Hz)', fontsize=11)
ax.set_ylabel('|Z| (Ω)', fontsize=11)
ax.set_title('(b) Bode Plot – Magnitude', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

# Phase on twin axis
ax2 = ax.twinx()
phase_true = np.degrees(np.angle(Z_true))
phase_fitted = np.degrees(np.angle(Z_fitted))
ax2.semilogx(freq, phase_true, 'b:', linewidth=1.5, alpha=0.6)
ax2.semilogx(freq, phase_fitted, 'r:', linewidth=1.5, alpha=0.6)
ax2.set_ylabel('Phase (°)', fontsize=10, color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

# --- (c) Parameter comparison bar chart ---
ax = axes[1, 0]
names = PARAM_ORDER
gt_vals_norm = []
fit_vals_norm = []
for i, name in enumerate(names):
    gt_v = gt_vec[i]
    fit_v = fitted_params[i]
    # Normalise to GT for visual comparison
    gt_vals_norm.append(1.0)
    fit_vals_norm.append(fit_v / gt_v)

x_pos = np.arange(len(names))
width = 0.35
bars1 = ax.bar(x_pos - width/2, gt_vals_norm, width, label='Ground Truth', color='steelblue')
bars2 = ax.bar(x_pos + width/2, fit_vals_norm, width, label='Fitted', color='salmon')
ax.set_xticks(x_pos)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel('Normalised Value (GT = 1)', fontsize=11)
ax.set_title('(c) Parameter Recovery (normalised)', fontsize=12)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax.legend(fontsize=9)
ax.grid(True, axis='y', alpha=0.3)

# Add relative error annotation
for i, name in enumerate(names):
    re = param_errors[name]['relative_error']
    ax.text(x_pos[i] + width/2, fit_vals_norm[i] + 0.02,
            f'{re:.1%}', ha='center', va='bottom', fontsize=8, color='red')

# --- (d) Residuals plot ---
ax = axes[1, 1]
residual_re = Z_fitted.real - Z_true.real
residual_im = Z_fitted.imag - Z_true.imag
ax.semilogx(freq, residual_re, 'b.-', markersize=4, label='ΔZ_real', alpha=0.7)
ax.semilogx(freq, residual_im, 'r.-', markersize=4, label='ΔZ_imag', alpha=0.7)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Frequency (Hz)', fontsize=11)
ax.set_ylabel('Residual (Ω)', fontsize=11)
ax.set_title('(d) Fit Residuals (Fitted − GT)', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('results/reconstruction_result.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✓ All results saved to results/")
print(f"  metrics.json, reconstruction_result.png, ground_truth.npy, reconstruction.npy")
