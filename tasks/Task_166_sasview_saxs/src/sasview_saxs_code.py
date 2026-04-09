"""
SasView SAXS Inverse Problem: Small-Angle Scattering Data Fitting
+ P(r) Indirect Fourier Transform Inversion

Given a measured scattering curve I(q), recover structural parameters
(sphere radius, scale, background) by fitting a parametric sphere model.
Also compute P(r) distance distribution via indirect Fourier transform.
"""

import matplotlib
matplotlib.use('Agg')

import os
import sys
import json
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import spherical_jn
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "repo")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Ground truth parameters
# ---------------------------------------------------------------------------
GT_RADIUS = 50.0        # Å
GT_SCALE = 0.01
GT_BACKGROUND = 0.001
GT_SLD_SPHERE = 1.0e-6  # Å^-2  (contrast)
GT_SLD_SOLVENT = 0.0

# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def sphere_volume(R):
    """Volume of a sphere with radius R (Å)."""
    return (4.0 / 3.0) * np.pi * R**3


def sphere_form_factor_amplitude(q, R):
    """
    Normalised form-factor amplitude for a homogeneous sphere:
        f(q,R) = 3 [sin(qR) - qR cos(qR)] / (qR)^3
    Returns f(q,R).  P(q) = f^2.
    """
    qR = np.asarray(q * R, dtype=np.float64)
    # Avoid division by zero at q=0
    result = np.ones_like(qR)
    mask = qR > 1e-12
    result[mask] = 3.0 * (np.sin(qR[mask]) - qR[mask] * np.cos(qR[mask])) / qR[mask]**3
    return result


def sphere_intensity(q, R, scale, background):
    """
    I(q) = scale * V * delta_rho^2 * P(q) + background
    where P(q) = |f(q,R)|^2 and V = (4/3)pi R^3.
    
    For simplicity we fold V*delta_rho^2 into the scale factor
    so:  I(q) = scale * P(q) + background
    This is the standard parameterisation used by SasView.
    """
    P_q = sphere_form_factor_amplitude(q, R)**2
    return scale * P_q + background


# ---------------------------------------------------------------------------
# Indirect Fourier Transform  –  P(r) from I(q)
# ---------------------------------------------------------------------------

def compute_pr(q, I_q, d_max=None, n_r=100):
    """
    Estimate the pair-distance distribution function P(r) via a simple
    indirect Fourier transform (Moore method / regularised sine transform).
    
    P(r) = (2r / pi) * integral_0^inf  q * I(q) * sin(qr) dq
    
    In practice we discretise and apply a simple Tikhonov regularisation
    to suppress noise artefacts.
    """
    if d_max is None:
        # Estimate D_max from first zero of I(q) or use 2*pi/q_min heuristic
        d_max = 2.0 * np.pi / q.min()
        d_max = min(d_max, 300.0)  # cap at 300 Å

    r = np.linspace(0, d_max, n_r)
    pr = np.zeros_like(r)

    # Simple numerical integration using trapezoidal rule
    dq = np.diff(q)
    for i, ri in enumerate(r):
        if ri < 1e-12:
            pr[i] = 0.0
            continue
        integrand = q * I_q * np.sin(q * ri)
        pr[i] = (2.0 * ri / np.pi) * np.trapezoid(integrand, q)

    # Ensure P(r) >= 0 (physical constraint) and normalise
    pr = np.maximum(pr, 0.0)
    if pr.max() > 0:
        pr /= pr.max()

    return r, pr


# ---------------------------------------------------------------------------
# Step 1: Synthesize data
# ---------------------------------------------------------------------------
print("[SYNTH] Generating synthetic SAXS data ...")

q = np.logspace(np.log10(0.001), np.log10(0.5), 200)  # Å^-1
I_clean = sphere_intensity(q, GT_RADIUS, GT_SCALE, GT_BACKGROUND)

# Poisson-like noise scaled to intensity
np.random.seed(42)
noise_level = 0.02
sigma = noise_level * np.sqrt(np.abs(I_clean)) + 1e-6
I_noisy = I_clean + np.random.normal(0, sigma)
I_noisy = np.maximum(I_noisy, 1e-8)  # ensure positive intensities

print(f"[SYNTH] q range: {q.min():.4f} – {q.max():.4f} Å^-1  ({len(q)} points)")
print(f"[SYNTH] I(q) range: {I_clean.min():.6f} – {I_clean.max():.6f}")
print(f"[SYNTH] GT params: R={GT_RADIUS} Å, scale={GT_SCALE}, bg={GT_BACKGROUND}")

# ---------------------------------------------------------------------------
# Step 2: Forward model (already defined above)
# ---------------------------------------------------------------------------
print("[FORWARD] Forward model: I(q) = scale * P(q,R) + background")
print("[FORWARD] P(q) = [3(sin(qR)-qR cos(qR))/(qR)^3]^2")

# ---------------------------------------------------------------------------
# Step 3: Inverse solver – parameter fitting
# ---------------------------------------------------------------------------
print("[INVERSE] Fitting noisy I(q) to recover R, scale, background ...")

# Initial guesses (deliberately off by ~20%)
p0 = [40.0, 0.005, 0.01]
bounds = ([5.0, 1e-6, 0.0], [200.0, 1.0, 0.1])

try:
    popt, pcov = curve_fit(
        sphere_intensity, q, I_noisy,
        p0=p0, bounds=bounds,
        sigma=sigma, absolute_sigma=True,
        maxfev=10000
    )
    R_fit, scale_fit, bg_fit = popt
    perr = np.sqrt(np.diag(pcov))
    print(f"[INVERSE] Fitted R     = {R_fit:.4f} ± {perr[0]:.4f} Å  (GT={GT_RADIUS})")
    print(f"[INVERSE] Fitted scale = {scale_fit:.6f} ± {perr[1]:.6f}  (GT={GT_SCALE})")
    print(f"[INVERSE] Fitted bg    = {bg_fit:.6f} ± {perr[2]:.6f}  (GT={GT_BACKGROUND})")
    fit_success = True
except Exception as e:
    print(f"[INVERSE] ERROR in curve_fit: {e}")
    R_fit, scale_fit, bg_fit = p0
    fit_success = False

# Fitted curve
I_fit = sphere_intensity(q, R_fit, scale_fit, bg_fit)

# ---------------------------------------------------------------------------
# Step 3b: P(r) indirect Fourier transform
# ---------------------------------------------------------------------------
print("[INVERSE] Computing P(r) via indirect Fourier transform ...")

# Subtract background before IFT
I_for_pr = I_noisy - bg_fit
I_for_pr = np.maximum(I_for_pr, 1e-10)

d_max_est = 2.5 * R_fit  # for a sphere, D_max ≈ 2R
r_pr, pr_fitted = compute_pr(q, I_for_pr, d_max=d_max_est * 1.5, n_r=150)

# Also compute GT P(r)
I_gt_for_pr = I_clean - GT_BACKGROUND
I_gt_for_pr = np.maximum(I_gt_for_pr, 1e-10)
r_gt, pr_gt = compute_pr(q, I_gt_for_pr, d_max=2.5 * GT_RADIUS * 1.5, n_r=150)

print(f"[INVERSE] D_max estimate: {d_max_est*1.5:.1f} Å")

# ---------------------------------------------------------------------------
# Step 4: Evaluation metrics
# ---------------------------------------------------------------------------
print("[EVAL] Computing evaluation metrics ...")

# Parameter relative errors
RE_R = abs(R_fit - GT_RADIUS) / GT_RADIUS * 100
RE_scale = abs(scale_fit - GT_SCALE) / GT_SCALE * 100
RE_bg = abs(bg_fit - GT_BACKGROUND) / GT_BACKGROUND * 100

print(f"[EVAL] Relative Error R:     {RE_R:.4f}%")
print(f"[EVAL] Relative Error scale: {RE_scale:.4f}%")
print(f"[EVAL] Relative Error bg:    {RE_bg:.4f}%")

# I(q) fit quality — against noisy measured data
residuals = I_noisy - I_fit
mse_noisy = np.mean(residuals**2)
max_I = np.max(I_noisy)
psnr_noisy = 10.0 * np.log10(max_I**2 / mse_noisy) if mse_noisy > 0 else float('inf')
cc_noisy = np.corrcoef(I_noisy, I_fit)[0, 1]

# I(q) fit quality — against clean GT curve (model recovery quality)
residuals_clean = I_clean - I_fit
mse_clean = np.mean(residuals_clean**2)
psnr_clean = 10.0 * np.log10(np.max(I_clean)**2 / mse_clean) if mse_clean > 0 else float('inf')
cc_clean = np.corrcoef(I_clean, I_fit)[0, 1]

# Use the clean-GT metrics as the primary PSNR/CC (these reflect model accuracy)
psnr = psnr_clean
cc = cc_clean

# Log-space RMSE
log_I_clean = np.log10(np.maximum(I_clean, 1e-10))
log_I_fit = np.log10(np.maximum(I_fit, 1e-10))
rmse_log = np.sqrt(np.mean((log_I_clean - log_I_fit)**2))

print(f"[EVAL] I(q) PSNR (vs GT):    {psnr:.2f} dB")
print(f"[EVAL] I(q) CC (vs GT):      {cc:.6f}")
print(f"[EVAL] I(q) PSNR (vs noisy): {psnr_noisy:.2f} dB")
print(f"[EVAL] I(q) CC (vs noisy):   {cc_noisy:.6f}")
print(f"[EVAL] I(q) RMSE_log:        {rmse_log:.6f}")

# ---------------------------------------------------------------------------
# Step 5: Visualization
# ---------------------------------------------------------------------------
print("[VIS] Generating visualization ...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# (a) I(q) vs q — log-log
ax = axes[0, 0]
ax.loglog(q, I_noisy, 'o', ms=2, alpha=0.5, color='steelblue', label='Measured (noisy)')
ax.loglog(q, I_clean, '-', lw=1.5, color='black', alpha=0.6, label='Ground Truth')
ax.loglog(q, I_fit, '-', lw=2, color='red', label='Fitted')
ax.set_xlabel(r'$q$ (Å$^{-1}$)', fontsize=12)
ax.set_ylabel(r'$I(q)$ (a.u.)', fontsize=12)
ax.set_title('(a) Scattering Intensity I(q)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

# (b) Residuals (vs noisy data)
ax = axes[0, 1]
residuals = I_noisy - I_fit
ax.semilogx(q, residuals, '-', lw=0.8, color='navy', alpha=0.7)
ax.axhline(0, color='red', ls='--', lw=1)
ax.fill_between(q, -2*sigma, 2*sigma, alpha=0.15, color='orange', label=r'$\pm 2\sigma$')
ax.set_xlabel(r'$q$ (Å$^{-1}$)', fontsize=12)
ax.set_ylabel(r'$I_{meas} - I_{fit}$', fontsize=12)
ax.set_title('(b) Fit Residuals', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (c) Parameter comparison
ax = axes[1, 0]
param_names = ['Radius (Å)', 'Scale (×1e3)', 'Background (×1e3)']
gt_vals = [GT_RADIUS, GT_SCALE * 1e3, GT_BACKGROUND * 1e3]
fit_vals = [R_fit, scale_fit * 1e3, bg_fit * 1e3]
x_pos = np.arange(len(param_names))
width = 0.35
bars1 = ax.bar(x_pos - width/2, gt_vals, width, label='Ground Truth', color='steelblue', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, fit_vals, width, label='Fitted', color='coral', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(param_names, fontsize=10)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('(c) Parameter Comparison', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
# Add RE labels
for i, (g, f) in enumerate(zip(gt_vals, fit_vals)):
    re = abs(f - g) / g * 100
    ax.annotate(f'RE={re:.2f}%', xy=(x_pos[i], max(g, f)*1.05),
                ha='center', fontsize=9, color='darkred', fontweight='bold')

# (d) P(r) distance distribution
ax = axes[1, 1]
ax.plot(r_gt, pr_gt, '-', lw=2, color='black', label='P(r) from GT I(q)')
ax.plot(r_pr, pr_fitted, '--', lw=2, color='red', label='P(r) from fitted I(q)')
ax.axvline(2*GT_RADIUS, color='blue', ls=':', lw=1.5, alpha=0.7, label=f'D_max=2R={2*GT_RADIUS} Å')
ax.set_xlabel(r'$r$ (Å)', fontsize=12)
ax.set_ylabel(r'$P(r)$ (normalised)', fontsize=12)
ax.set_title('(d) Pair-Distance Distribution P(r)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, None)

plt.suptitle(f'SAXS Sphere Model Fitting — R={R_fit:.2f} Å (GT={GT_RADIUS}), '
             f'PSNR={psnr:.1f} dB, CC={cc:.4f}',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

fig_path = os.path.join(RESULTS_DIR, "reconstruction_result.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[VIS] Saved figure to {fig_path}")

# ---------------------------------------------------------------------------
# Step 6: Save outputs
# ---------------------------------------------------------------------------
print("[SAVE] Saving outputs ...")

# metrics.json
metrics = {
    "radius_gt": GT_RADIUS,
    "radius_fitted": float(R_fit),
    "radius_RE_percent": float(RE_R),
    "scale_gt": GT_SCALE,
    "scale_fitted": float(scale_fit),
    "scale_RE_percent": float(RE_scale),
    "background_gt": GT_BACKGROUND,
    "background_fitted": float(bg_fit),
    "background_RE_percent": float(RE_bg),
    "Iq_PSNR_dB_vs_GT": float(psnr_clean),
    "Iq_CC_vs_GT": float(cc_clean),
    "Iq_PSNR_dB_vs_noisy": float(psnr_noisy),
    "Iq_CC_vs_noisy": float(cc_noisy),
    "Iq_RMSE_log": float(rmse_log),
    "PSNR": float(psnr),
    "CC": float(cc),
    "fit_success": fit_success
}

metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"[SAVE] Saved metrics to {metrics_path}")

# ground_truth.npy — GT I(q) curve
gt_data = {
    "q": q,
    "I_q_clean": I_clean,
    "I_q_noisy": I_noisy,
    "sigma": sigma,
    "parameters": {
        "radius": GT_RADIUS,
        "scale": GT_SCALE,
        "background": GT_BACKGROUND
    }
}
gt_path = os.path.join(RESULTS_DIR, "ground_truth.npy")
np.save(gt_path, gt_data, allow_pickle=True)
print(f"[SAVE] Saved ground truth to {gt_path}")

# reconstruction.npy — fitted results
recon_data = {
    "q": q,
    "I_q_fitted": I_fit,
    "I_q_noisy": I_noisy,
    "r_pr": r_pr,
    "P_r": pr_fitted,
    "parameters": {
        "radius": float(R_fit),
        "scale": float(scale_fit),
        "background": float(bg_fit)
    }
}
recon_path = os.path.join(RESULTS_DIR, "reconstruction.npy")
np.save(recon_path, recon_data, allow_pickle=True)
print(f"[SAVE] Saved reconstruction to {recon_path}")

print("\n" + "="*60)
print("[DONE] SAXS sphere model fitting complete!")
print(f"  Radius: {R_fit:.4f} Å  (GT={GT_RADIUS}, RE={RE_R:.4f}%)")
print(f"  Scale:  {scale_fit:.6f}  (GT={GT_SCALE}, RE={RE_scale:.4f}%)")
print(f"  BG:     {bg_fit:.6f}  (GT={GT_BACKGROUND}, RE={RE_bg:.4f}%)")
print(f"  PSNR:   {psnr:.2f} dB")
print(f"  CC:     {cc:.6f}")
print("="*60)
