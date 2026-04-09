"""
thermoelastic — Thermoelastic Stress Analysis
================================================
From infrared surface temperature measurements under cyclic loading,
reconstruct the stress field using the thermoelastic effect:
    ΔT = −(α T₀)/(ρ Cp) × Δ(σ₁ + σ₂)

Ground truth: Kirsch solution for a plate with a central hole under uniaxial
tension.

Inverse: Direct algebraic inversion of the thermoelastic equation, then
comparison with the known stress sum field.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json
from skimage.metrics import structural_similarity as ssim

# ── paths ──────────────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR  = "/data/yjh/website_assets/Task_116_thermoelastic"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── material constants (aluminium) ────────────────────────────────
ALPHA   = 23e-6      # CTE (1/K)
T0      = 293.0      # reference temperature (K)
RHO     = 2700.0     # density (kg/m³)
CP      = 900.0      # specific heat (J/(kg·K))
THERMO_COEFF = ALPHA * T0 / (RHO * CP)   # ≈ 2.77e-9  K/Pa

# ── geometry / loading ─────────────────────────────────────────────
SIGMA_0 = 100e6      # far-field stress (Pa) = 100 MPa
A_HOLE  = 0.01       # hole radius (m) = 10 mm
PLATE_R = 0.05       # outer radius for grid (m) = 50 mm
NR      = 200        # radial grid points
NTHETA  = 360        # angular grid points

# ── noise ──────────────────────────────────────────────────────────
NOISE_LEVEL = 0.03   # 3 % of max |ΔT|
SEED = 42
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════
# 1.  GROUND TRUTH — Kirsch Solution
# ═══════════════════════════════════════════════════════════════════
def kirsch_stress(r, theta, sigma_0, a):
    """
    Kirsch solution for an infinite plate with a circular hole of
    radius *a* under uniaxial tension σ₀ in the x-direction.

    Returns σ_rr, σ_θθ, σ_rθ in polar coordinates.
    """
    ratio  = a / r
    ratio2 = ratio ** 2
    ratio4 = ratio ** 4
    cos2t  = np.cos(2 * theta)

    sigma_rr  = (sigma_0 / 2) * ((1 - ratio2) + (1 - 4 * ratio2 + 3 * ratio4) * cos2t)
    sigma_tt  = (sigma_0 / 2) * ((1 + ratio2) - (1 + 3 * ratio4) * cos2t)
    return sigma_rr, sigma_tt


def stress_sum_field(r, theta, sigma_0, a):
    """σ₁ + σ₂ = σ_rr + σ_θθ  (sum of principal stresses in 2D)."""
    sigma_rr, sigma_tt = kirsch_stress(r, theta, sigma_0, a)
    return sigma_rr + sigma_tt


# ═══════════════════════════════════════════════════════════════════
# 2.  FORWARD:  Stress → Temperature Change
# ═══════════════════════════════════════════════════════════════════
def stress_to_temperature(stress_sum):
    """
    Thermoelastic equation:
        ΔT = −(α T₀)/(ρ Cp) × Δ(σ₁ + σ₂)
    """
    return -THERMO_COEFF * stress_sum


# ═══════════════════════════════════════════════════════════════════
# 3.  INVERSE:  Temperature → Stress Sum
# ═══════════════════════════════════════════════════════════════════
def temperature_to_stress(delta_T):
    """
    Direct inversion:
        Δ(σ₁ + σ₂) = −(ρ Cp)/(α T₀) × ΔT
    """
    return -delta_T / THERMO_COEFF


# ═══════════════════════════════════════════════════════════════════
# 4.  METRICS
# ═══════════════════════════════════════════════════════════════════
def compute_metrics(gt_field, recon_field):
    """PSNR, SSIM, RMSE, CC on the 2D stress-sum field."""
    mse = np.mean((gt_field - recon_field) ** 2)
    data_range = np.max(gt_field) - np.min(gt_field)
    psnr = 10.0 * np.log10(data_range ** 2 / (mse + 1e-30))

    cc = float(np.corrcoef(gt_field.ravel(), recon_field.ravel())[0, 1])
    rmse = float(np.sqrt(mse))

    # SSIM — normalise to [0, 1]
    gt_n = (gt_field - gt_field.min()) / (gt_field.max() - gt_field.min() + 1e-30)
    rc_n = (recon_field - recon_field.min()) / (recon_field.max() - recon_field.min() + 1e-30)
    ssim_val = float(ssim(gt_n, rc_n, data_range=1.0))

    return {"PSNR": float(psnr), "SSIM": ssim_val, "CC": cc,
            "RMSE": rmse, "RMSE_MPa": rmse / 1e6}


# ═══════════════════════════════════════════════════════════════════
# 5.  VISUALISATION
# ═══════════════════════════════════════════════════════════════════
def visualize(R, THETA, gt_sum, delta_T_noisy, recon_sum, metrics):
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) GT stress sum
    ax = axes[0, 0]
    im = ax.pcolormesh(X * 1e3, Y * 1e3, gt_sum / 1e6, cmap="RdBu_r", shading="auto")
    ax.set_title("GT Stress Sum  σ₁+σ₂  (MPa)")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)

    # (b) Temperature change
    ax = axes[0, 1]
    im = ax.pcolormesh(X * 1e3, Y * 1e3, delta_T_noisy * 1e3, cmap="coolwarm", shading="auto")
    ax.set_title("Measured ΔT (mK)")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)

    # (c) Recovered stress sum
    ax = axes[1, 0]
    im = ax.pcolormesh(X * 1e3, Y * 1e3, recon_sum / 1e6, cmap="RdBu_r", shading="auto")
    ax.set_title(f"Recovered σ₁+σ₂  (PSNR={metrics['PSNR']:.1f} dB)")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)

    # (d) Error
    ax = axes[1, 1]
    err = (gt_sum - recon_sum) / 1e6
    im = ax.pcolormesh(X * 1e3, Y * 1e3, err, cmap="bwr", shading="auto")
    ax.set_title(f"Error  (RMSE={metrics['RMSE_MPa']:.2f} MPa, CC={metrics['CC']:.4f})")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)

    plt.suptitle("Thermoelastic Stress Analysis — Plate with Hole", fontsize=14, y=1.02)
    plt.tight_layout()
    for p in [os.path.join(RESULTS_DIR, "reconstruction_result.png"),
              os.path.join(ASSETS_DIR,  "reconstruction_result.png"),
              os.path.join(ASSETS_DIR,  "vis_result.png")]:
        plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# 6.  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("thermoelastic — Thermoelastic Stress Analysis")
    print("=" * 60)

    # 1. Polar grid (exclude hole interior)
    r     = np.linspace(A_HOLE * 1.01, PLATE_R, NR)
    theta = np.linspace(0, 2 * np.pi, NTHETA, endpoint=False)
    R, THETA = np.meshgrid(r, theta, indexing="ij")

    # 2. GT stress sum
    print("[1/4] Computing Kirsch stress field ...")
    gt_sum = stress_sum_field(R, THETA, SIGMA_0, A_HOLE)

    # 3. Forward: temperature change
    print("[2/4] Computing thermoelastic temperature change ...")
    delta_T_clean = stress_to_temperature(gt_sum)
    noise = NOISE_LEVEL * np.max(np.abs(delta_T_clean)) * np.random.randn(*delta_T_clean.shape)
    delta_T_noisy = delta_T_clean + noise

    # 4. Inverse
    print("[3/4] Inverting temperature to stress ...")
    recon_sum = temperature_to_stress(delta_T_noisy)

    # 5. Metrics
    metrics = compute_metrics(gt_sum, recon_sum)
    print(f"  PSNR = {metrics['PSNR']:.2f} dB")
    print(f"  SSIM = {metrics['SSIM']:.4f}")
    print(f"  CC   = {metrics['CC']:.6f}")
    print(f"  RMSE = {metrics['RMSE_MPa']:.4f} MPa")

    # 6. Save
    print("[4/4] Saving results ...")
    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), gt_sum)
        np.save(os.path.join(d, "recon_output.npy"), recon_sum)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    visualize(R, THETA, gt_sum, delta_T_noisy, recon_sum, metrics)

    print("Done ✓")
    return metrics


if __name__ == "__main__":
    main()
