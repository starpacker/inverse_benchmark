"""
Task 142: phasorpy_flim — Phasor-based FLIM lifetime component analysis.

Inverse problem: From frequency-domain FLIM data, extract fluorescence
lifetime component fractions using phasor analysis (lever rule / linear
unmixing on the phasor plot).

Forward model
-------------
Create a 128×128 synthetic FLIM image with 2 fluorophore species at known
lifetimes (τ₁=1.0 ns, τ₂=4.0 ns) with a spatially varying fraction map.
Generate time-resolved fluorescence decay at each pixel and add Poisson noise.

Inverse
-------
Compute phasor coordinates (G, S) from the time-domain data using phasorpy,
then decompose each pixel's phasor into fractions of the two known lifetime
components using the lever rule (two_fractions_from_phasor).

Metrics: PSNR, SSIM of the recovered fraction-1 map vs ground truth.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from phasorpy.components import two_fractions_from_phasor
from phasorpy.phasor import (
    phasor_from_lifetime,
    phasor_from_signal,
    phasor_semicircle,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
NX, NY = 128, 128          # image size
N_TIME = 256                # number of time bins
TAU1_NS = 1.0               # lifetime of species 1 (ns)
TAU2_NS = 4.0               # lifetime of species 2 (ns)
FREQ_MHZ = 80.0             # laser repetition frequency (MHz)
TOTAL_PHOTONS = 5000         # mean total photon count per pixel
RNG_SEED = 42
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUTDIR, exist_ok=True)

rng = np.random.default_rng(RNG_SEED)

# ---------------------------------------------------------------------------
# 1. Ground-truth fraction map  (species-1 fraction varies spatially)
# ---------------------------------------------------------------------------
# Smooth gradient + circular region to make it interesting
yy, xx = np.meshgrid(np.linspace(0, 1, NY), np.linspace(0, 1, NX))
# Base: horizontal gradient
f1_gt = 0.2 + 0.6 * xx  # ranges from 0.2 to 0.8

# Add a circular region of high species-2 (low f1)
cx, cy, r = 0.65, 0.35, 0.18
mask_circle = ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2
f1_gt[mask_circle] = 0.15

# Add a square region of high species-1
f1_gt[20:45, 80:105] = 0.90

f1_gt = np.clip(f1_gt, 0.0, 1.0).astype(np.float64)
f2_gt = 1.0 - f1_gt

print(f"Ground-truth fraction map: f1 range [{f1_gt.min():.3f}, {f1_gt.max():.3f}]")

# ---------------------------------------------------------------------------
# 2. Forward model — generate time-domain FLIM data
# ---------------------------------------------------------------------------
# Time axis (one laser period)
period_ns = 1e3 / FREQ_MHZ  # 12.5 ns for 80 MHz
t = np.linspace(0, period_ns, N_TIME, endpoint=False)  # (N_TIME,)
dt = t[1] - t[0]

# Exponential decays for each species (normalized to integrate to 1)
decay1 = np.exp(-t / TAU1_NS)
decay1 /= decay1.sum()

decay2 = np.exp(-t / TAU2_NS)
decay2 /= decay2.sum()

# Per-pixel decay = f1*decay1 + f2*decay2, scaled by total photon count
# shape: (NX, NY, N_TIME)
clean_signal = (
    f1_gt[:, :, np.newaxis] * decay1[np.newaxis, np.newaxis, :]
    + f2_gt[:, :, np.newaxis] * decay2[np.newaxis, np.newaxis, :]
) * TOTAL_PHOTONS

# Add Poisson noise
noisy_signal = rng.poisson(np.maximum(clean_signal, 1e-12)).astype(np.float64)

print(f"FLIM data shape: {noisy_signal.shape}, mean counts/pixel: {noisy_signal.sum(axis=-1).mean():.0f}")

# ---------------------------------------------------------------------------
# 3. Phasor transform — compute (G, S) for each pixel
# ---------------------------------------------------------------------------
# phasor_from_signal expects signal along axis; returns (mean, real, imag)
# axis=-1 (last axis = time)
mean_intensity, G_meas, S_meas = phasor_from_signal(
    noisy_signal, axis=-1, harmonic=1
)

print(f"Phasor coords: G range [{G_meas.min():.4f}, {G_meas.max():.4f}], "
      f"S range [{S_meas.min():.4f}, {S_meas.max():.4f}]")

# ---------------------------------------------------------------------------
# 4. Compute known component phasor positions on the semicircle
# ---------------------------------------------------------------------------
G_ref, S_ref = phasor_from_lifetime(FREQ_MHZ, np.array([TAU1_NS, TAU2_NS]))
print(f"Component phasors: τ₁={TAU1_NS} ns → G={G_ref[0]:.4f}, S={S_ref[0]:.4f}")
print(f"                   τ₂={TAU2_NS} ns → G={G_ref[1]:.4f}, S={S_ref[1]:.4f}")

# ---------------------------------------------------------------------------
# 5. Inverse — linear unmixing via lever rule
# ---------------------------------------------------------------------------
f1_recon = two_fractions_from_phasor(
    G_meas, S_meas,
    G_ref, S_ref,
)
f1_recon = np.clip(f1_recon, 0.0, 1.0)

print(f"Recovered f1 range [{f1_recon.min():.4f}, {f1_recon.max():.4f}]")

# ---------------------------------------------------------------------------
# 6. Metrics
# ---------------------------------------------------------------------------
# Data range for PSNR (fraction is [0, 1])
psnr_val = psnr(f1_gt, f1_recon, data_range=1.0)
ssim_val = ssim(f1_gt, f1_recon, data_range=1.0)

# Lifetime-related relative errors: compute mean recovered lifetime per pixel
# τ_eff = f1*τ1 + f2*τ2
tau_eff_gt = f1_gt * TAU1_NS + f2_gt * TAU2_NS
tau_eff_recon = f1_recon * TAU1_NS + (1 - f1_recon) * TAU2_NS
tau_re = np.mean(np.abs(tau_eff_gt - tau_eff_recon) / tau_eff_gt)

# Mean absolute error of fraction
mae_f1 = np.mean(np.abs(f1_gt - f1_recon))

metrics = {
    "PSNR_dB": round(float(psnr_val), 2),
    "SSIM": round(float(ssim_val), 6),
    "fraction_MAE": round(float(mae_f1), 6),
    "lifetime_eff_RE": round(float(tau_re), 6),
    "tau1_ns": TAU1_NS,
    "tau2_ns": TAU2_NS,
    "frequency_MHz": FREQ_MHZ,
    "image_size": [NX, NY],
    "total_photons_per_pixel": TOTAL_PHOTONS,
}

print("\n=== Metrics ===")
for k, v in metrics.items():
    print(f"  {k}: {v}")

# Save metrics
with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# Save arrays
np.save(os.path.join(OUTDIR, "ground_truth.npy"), f1_gt)
np.save(os.path.join(OUTDIR, "recon_output.npy"), f1_recon)

print("\nSaved ground_truth.npy, recon_output.npy, metrics.json")

# ---------------------------------------------------------------------------
# 7. Visualization — 4-panel figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 11))

# Panel 1: GT fraction map
im0 = axes[0, 0].imshow(f1_gt, cmap="viridis", vmin=0, vmax=1, origin="lower")
axes[0, 0].set_title("Ground Truth: Species-1 Fraction", fontsize=12)
plt.colorbar(im0, ax=axes[0, 0], label="$f_1$")

# Panel 2: Phasor plot
ax_ph = axes[0, 1]
# Draw universal semicircle
sc_g, sc_s = phasor_semicircle()
ax_ph.plot(sc_g, sc_s, "k-", linewidth=1.5, label="Universal semicircle")
# Scatter all pixel phasors (subsample for clarity)
step = max(1, NX * NY // 3000)
g_flat = G_meas.ravel()[::step]
s_flat = S_meas.ravel()[::step]
f1_flat = f1_gt.ravel()[::step]
sc = ax_ph.scatter(g_flat, s_flat, c=f1_flat, cmap="viridis", s=3, alpha=0.5,
                   vmin=0, vmax=1)
plt.colorbar(sc, ax=ax_ph, label="$f_1$ (GT)")
# Mark component positions
ax_ph.plot(G_ref[0], S_ref[0], "r^", markersize=12, label=f"τ₁={TAU1_NS} ns")
ax_ph.plot(G_ref[1], S_ref[1], "bs", markersize=12, label=f"τ₂={TAU2_NS} ns")
ax_ph.set_xlabel("G (real)", fontsize=11)
ax_ph.set_ylabel("S (imaginary)", fontsize=11)
ax_ph.set_title("Phasor Plot", fontsize=12)
ax_ph.set_xlim(-0.05, 1.05)
ax_ph.set_ylim(-0.05, 0.6)
ax_ph.set_aspect("equal")
ax_ph.legend(fontsize=9, loc="upper right")

# Panel 3: Reconstructed fraction map
im2 = axes[1, 0].imshow(f1_recon, cmap="viridis", vmin=0, vmax=1, origin="lower")
axes[1, 0].set_title(
    f"Reconstructed: Species-1 Fraction\nPSNR={psnr_val:.1f} dB, SSIM={ssim_val:.4f}",
    fontsize=12,
)
plt.colorbar(im2, ax=axes[1, 0], label="$f_1$")

# Panel 4: Error map
error = f1_recon - f1_gt
emax = max(abs(error.min()), abs(error.max()), 0.05)
im3 = axes[1, 1].imshow(error, cmap="RdBu_r", vmin=-emax, vmax=emax, origin="lower")
axes[1, 1].set_title(f"Error (Recon − GT), MAE={mae_f1:.4f}", fontsize=12)
plt.colorbar(im3, ax=axes[1, 1], label="$\\Delta f_1$")

fig.suptitle(
    "Task 142: Phasor-based FLIM Lifetime Component Analysis\n"
    f"τ₁={TAU1_NS} ns, τ₂={TAU2_NS} ns, freq={FREQ_MHZ} MHz, "
    f"{TOTAL_PHOTONS} photons/px",
    fontsize=14,
    fontweight="bold",
    y=0.99,
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
figpath = os.path.join(OUTDIR, "reconstruction_result.png")
fig.savefig(figpath, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {figpath}")

print("\n✓ Task 142 (phasorpy_flim) complete.")
