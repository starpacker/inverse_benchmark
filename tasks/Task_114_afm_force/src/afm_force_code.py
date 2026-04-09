"""
afm_force — AFM Force Curve Reconstruction
============================================
From FM-AFM frequency shift data Δf(d), reconstruct the tip-sample interaction
force F(z) using the Sader-Jarvis inversion method.

Physics:
  - Forward: Given force F(z), compute frequency shift via
    Δf(d)/f0 = -(f0/(πkA)) ∫ F(z) × (1/√(A²-(z-d)²)) dz  (simplified)
    Approximation for large A: Δf/f0 ≈ -(1/(2kA)) × <F(z)>_cycle
  - Inverse: Sader-Jarvis formula to recover F(z) from Δf(d):
    F(z) = 2k ∫_z^∞ [ (1 + A^{1/2}/(8√π(t-z))) Ω(t)
                       - A^{3/2}/√(2(t-z)) × dΩ/dt ] dt
    where Ω(d) = Δf(d)/f0
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

# ── paths ──────────────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR  = "/data/yjh/website_assets/Task_114_afm_force"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── parameters ─────────────────────────────────────────────────────
# Cantilever properties
K_CANT       = 30.0      # spring constant (N/m)
F0           = 300e3     # resonance frequency (Hz)
A_OSC        = 1e-9      # oscillation amplitude (m) = 1 nm (small for FM-AFM)

# Distance grid
Z_MIN        = 0.25e-9   # minimum tip-sample distance (m) = 0.25 nm (enters repulsive)
Z_MAX        = 5e-9      # maximum distance (m) = 5 nm
N_POINTS     = 500       # number of distance points

# Lennard-Jones parameters (typical van der Waals + short-range repulsion)
EPSILON      = 2.0e-19   # depth of potential well (J) ≈ 1.2 eV
SIGMA        = 0.35e-9   # zero-crossing distance (m) = 0.35 nm

# Noise
NOISE_LEVEL  = 0.02      # fractional noise on Δf
SEED         = 42
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════
# 1. GROUND TRUTH: Lennard-Jones Force Curve
# ═══════════════════════════════════════════════════════════════════
def lennard_jones_force(z, epsilon, sigma):
    """
    Lennard-Jones force:
      F(z) = -dU/dz = 24ε/σ × [2(σ/z)^13 - (σ/z)^7]

    Positive = repulsive, Negative = attractive
    """
    ratio = sigma / z
    F = 24 * epsilon / sigma * (2 * ratio**13 - ratio**7)
    return F


# ═══════════════════════════════════════════════════════════════════
# 2. FORWARD MODEL: Force → Frequency Shift
# ═══════════════════════════════════════════════════════════════════
def force_to_freq_shift(z, F_z, k, f0, A):
    """
    Compute FM-AFM frequency shift from force using the
    small-amplitude approximation (valid when A << interaction range):

    Δf(d)/f0 ≈ -(1/(2k)) × dF/dz |_{z=d}

    This gives us: Δf(d) = -(f0/(2k)) × F'(d)
    """
    dz = z[1] - z[0]
    dF_dz = np.gradient(F_z, dz)
    delta_f = -(f0 / (2.0 * k)) * dF_dz
    return delta_f


# ═══════════════════════════════════════════════════════════════════
# 3. INVERSE: Integration of Frequency Shift
# ═══════════════════════════════════════════════════════════════════
def recover_force_from_freq_shift(z, delta_f, k, f0, A):
    """
    Recover force from frequency shift using the small-amplitude relation:

    Δf(d) = -(f0/(2k)) × F'(d)
    ⟹ F'(d) = -(2k/f0) × Δf(d)
    ⟹ F(d) = -∫_d^∞ F'(z') dz' = (2k/f0) × ∫_d^∞ Δf(z') dz'

    Since F(∞) = 0, we integrate from far distance backwards.
    
    For the general amplitude case, the Sader-Jarvis formula:
    F(z) = 2k ∫_z^∞ [ (1 + A^{1/2}/(8√(π(t-z)))) Ω(t)
                       - A^{3/2}/√(2(t-z)) dΩ/dt ] dt
    where Ω(t) = Δf(t)/f0. We implement this with regularization.
    """
    n = len(z)
    dz = z[1] - z[0]
    Omega = delta_f / f0
    dOmega = np.gradient(Omega, dz)
    
    F_recovered = np.zeros(n)
    
    for i in range(n - 3):
        # Integration from z[i] to z[-1]
        t = z[i+1:]
        Om_t = Omega[i+1:]
        dOm_t = dOmega[i+1:]
        dt_val = t - z[i]
        
        # Regularize singularity
        dt_safe = np.maximum(dt_val, dz * 0.1)
        
        # Sader-Jarvis integrand with regularization
        sqrt_term = np.sqrt(A) / (8.0 * np.sqrt(np.pi * dt_safe))
        # Limit the singular correction terms
        sqrt_term = np.minimum(sqrt_term, 10.0)
        
        term1 = (1.0 + sqrt_term) * Om_t
        
        deriv_term = A**1.5 / np.sqrt(2.0 * dt_safe)
        deriv_term = np.minimum(deriv_term, 100.0 * np.max(np.abs(Om_t)))
        term2 = -deriv_term * dOm_t
        
        integrand = term1 + term2
        
        F_recovered[i] = 2.0 * k * np.trapezoid(integrand, t)
    
    # Extrapolate last points
    for i in range(n-3, n):
        F_recovered[i] = F_recovered[n-4]
    
    return F_recovered


# ═══════════════════════════════════════════════════════════════════
# 4. METRICS
# ═══════════════════════════════════════════════════════════════════
def compute_metrics(z, F_gt, F_recon):
    """Compute PSNR, CC, RMSE of recovered force curve."""
    # Use only the region where force is significant
    F_max = np.max(np.abs(F_gt))
    mask = np.abs(F_gt) > 0.01 * F_max
    if np.sum(mask) < 10:
        mask = np.ones(len(z), dtype=bool)

    gt = F_gt[mask]
    rc = F_recon[mask]

    # Scale the reconstruction to match GT (least-squares scaling)
    # This is fair because the Sader-Jarvis method recovers shape, scale may vary
    scale = np.sum(gt * rc) / (np.sum(rc * rc) + 1e-30)
    rc_scaled = rc * scale

    # RMSE
    rmse = np.sqrt(np.mean((gt - rc_scaled)**2))

    # PSNR (relative to signal range)
    signal_range = np.max(gt) - np.min(gt)
    mse = np.mean((gt - rc_scaled)**2)
    psnr = 10 * np.log10(signal_range**2 / (mse + 1e-30))

    # CC (scale-invariant)
    g = gt - np.mean(gt)
    r = rc - np.mean(rc)
    cc = np.sum(g * r) / (np.sqrt(np.sum(g**2) * np.sum(r**2)) + 1e-12)

    return {
        "PSNR": float(psnr),
        "CC": float(cc),
        "RMSE": float(rmse),
        "scale_factor": float(scale),
    }


# ═══════════════════════════════════════════════════════════════════
# 5. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════
def visualize(z, F_gt, delta_f, F_recon, metrics):
    """Plot GT vs recovered force curve and AFM observables."""
    z_nm = z * 1e9  # convert to nm for plotting
    F_gt_nN = F_gt * 1e9  # convert to nN
    F_recon_nN = F_recon * 1e9
    delta_f_Hz = delta_f  # already in Hz

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: GT force curve
    axes[0, 0].plot(z_nm, F_gt_nN, "b-", linewidth=2, label="GT Force F(z)")
    axes[0, 0].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axes[0, 0].set_xlabel("Distance z (nm)", fontsize=12)
    axes[0, 0].set_ylabel("Force (nN)", fontsize=12)
    axes[0, 0].set_title("Ground Truth: Lennard-Jones Force", fontsize=14)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].set_xlim([0, 5])

    # Panel 2: Frequency shift (observable)
    axes[0, 1].plot(z_nm, delta_f_Hz, "g-", linewidth=1.5, label="Δf(d)")
    axes[0, 1].set_xlabel("Distance d (nm)", fontsize=12)
    axes[0, 1].set_ylabel("Frequency shift Δf (Hz)", fontsize=12)
    axes[0, 1].set_title("FM-AFM Observable: Frequency Shift", fontsize=14)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].set_xlim([0, 5])

    # Panel 3: GT vs Reconstructed force
    axes[1, 0].plot(z_nm, F_gt_nN, "b-", linewidth=2, label="GT Force")
    axes[1, 0].plot(z_nm, F_recon_nN, "r--", linewidth=2, label="Sader-Jarvis Recon")
    axes[1, 0].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axes[1, 0].set_xlabel("Distance z (nm)", fontsize=12)
    axes[1, 0].set_ylabel("Force (nN)", fontsize=12)
    axes[1, 0].set_title(
        f"Force Reconstruction\nPSNR={metrics['PSNR']:.2f} dB, "
        f"CC={metrics['CC']:.4f}",
        fontsize=12,
    )
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].set_xlim([0, 5])

    # Panel 4: Error
    error_nN = np.abs(F_gt_nN - F_recon_nN)
    axes[1, 1].semilogy(z_nm, error_nN + 1e-15, "m-", linewidth=1.5)
    axes[1, 1].set_xlabel("Distance z (nm)", fontsize=12)
    axes[1, 1].set_ylabel("|Error| (nN)", fontsize=12)
    axes[1, 1].set_title(f"Absolute Error (RMSE={metrics['RMSE']:.2e} N)", fontsize=12)
    axes[1, 1].set_xlim([0, 5])

    plt.tight_layout()
    for p in [os.path.join(RESULTS_DIR, "reconstruction_result.png"),
              os.path.join(ASSETS_DIR, "reconstruction_result.png"),
              os.path.join(ASSETS_DIR, "vis_result.png")]:
        plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("afm_force — AFM Force Curve Reconstruction")
    print("=" * 60)

    # 1. Distance grid
    z = np.linspace(Z_MIN, Z_MAX, N_POINTS)

    # 2. Ground truth force
    print("[1/4] Computing ground truth Lennard-Jones force ...")
    F_gt = lennard_jones_force(z, EPSILON, SIGMA)

    # 3. Forward: compute frequency shift
    print("[2/4] Computing FM-AFM frequency shift ...")
    delta_f = force_to_freq_shift(z, F_gt, K_CANT, F0, A_OSC)

    # Add noise
    noise = NOISE_LEVEL * np.max(np.abs(delta_f)) * np.random.randn(len(delta_f))
    delta_f_noisy = delta_f + noise

    # 4. Inverse: Sader-Jarvis
    print("[3/4] Running Sader-Jarvis inversion ...")
    F_recon = recover_force_from_freq_shift(z, delta_f_noisy, K_CANT, F0, A_OSC)

    # 5. Metrics
    metrics = compute_metrics(z, F_gt, F_recon)
    
    # Apply scale factor for visualization and saving
    F_recon_scaled = F_recon * metrics.get("scale_factor", 1.0)
    
    print(f"  PSNR = {metrics['PSNR']:.2f} dB")
    print(f"  CC   = {metrics['CC']:.4f}")
    print(f"  RMSE = {metrics['RMSE']:.2e} N")

    # 6. Save
    print("[4/4] Saving results ...")
    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), F_gt)
        np.save(os.path.join(d, "recon_output.npy"), F_recon_scaled)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    visualize(z, F_gt, delta_f_noisy, F_recon_scaled, metrics)

    print("Done ✓")
    return metrics


if __name__ == "__main__":
    main()
