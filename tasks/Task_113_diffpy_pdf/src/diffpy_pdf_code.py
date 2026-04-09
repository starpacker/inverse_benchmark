"""
diffpy_pdf — Pair Distribution Function Analysis
==================================================
From powder diffraction data, extract the atomic pair distribution function G(r)
and fit structural parameters (lattice constant, atomic displacement, coordination).

Physics:
  - Forward: PDF from structure
    G(r) = Σ_n A_n × Gaussian(r - d_n, σ_n)
    Peak positions correspond to interatomic distances in the crystal.
    For FCC: d_n = a × {1/√2, 1, √(3/2), √2, √(5/2), √3, ...}
  - Inverse: Least-squares fitting of structural parameters
    (lattice constant a, Debye-Waller factor B, scale)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
from skimage.metrics import structural_similarity as ssim

# ── paths ──────────────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR  = "/data/yjh/website_assets/Task_113_diffpy_pdf"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── parameters ─────────────────────────────────────────────────────
R_MIN        = 1.0       # Å
R_MAX        = 30.0      # Å
DR           = 0.01      # Å step
NOISE_LEVEL  = 0.05      # fractional noise on G(r)
SEED         = 42
np.random.seed(SEED)

# Ground truth FCC copper parameters
A_TRUE       = 3.615     # lattice constant (Å) for Cu
B_TRUE       = 0.55      # Debye-Waller factor (Å²)
SCALE_TRUE   = 1.0       # overall scale
RHO0_TRUE    = 4 / A_TRUE**3   # number density for FCC (4 atoms/cell)


# ═══════════════════════════════════════════════════════════════════
# 1. FCC neighbor distances
# ═══════════════════════════════════════════════════════════════════
def fcc_neighbor_distances(a, r_max, max_shell=200):
    """
    Compute interatomic distances and coordination numbers for FCC structure.

    For FCC, neighbor distances are a × sqrt(n/2) for certain n values.
    Returns list of (distance, coordination_number) pairs.
    """
    distances = []
    # Generate all lattice vectors and compute distances
    n_max = int(np.ceil(r_max / a)) + 1
    for h in range(-n_max, n_max + 1):
        for k in range(-n_max, n_max + 1):
            for l in range(-n_max, n_max + 1):
                # FCC basis positions: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
                for bx, by, bz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    x = (h + bx) * a
                    y = (k + by) * a
                    z = (l + bz) * a
                    d = np.sqrt(x**2 + y**2 + z**2)
                    if 0.1 < d < r_max:
                        distances.append(d)

    # Bin into shells
    distances = np.sort(distances)
    shells = []
    tol = 0.01
    i = 0
    while i < len(distances) and len(shells) < max_shell:
        d_ref = distances[i]
        count = 0
        while i < len(distances) and abs(distances[i] - d_ref) < tol:
            count += 1
            i += 1
        shells.append((d_ref, count))

    return shells


# ═══════════════════════════════════════════════════════════════════
# 2. FORWARD MODEL: Compute PDF G(r)
# ═══════════════════════════════════════════════════════════════════
def compute_pdf(r, a, B, scale, r_max=30.0):
    """
    Compute the reduced pair distribution function G(r) for FCC.

    G(r) = (1/r) Σ_n [N_n / (4πr_n²ρ₀)] × Gaussian(r - r_n, σ_n) - 4πrρ₀

    Simplified as sum of Gaussian peaks:
    G(r) = scale × Σ_n C_n × exp(-(r - d_n)² / (2σ_n²)) / (σ_n√(2π) × r)

    where σ_n² = B (isotropic Debye-Waller), C_n = coordination number
    """
    shells = fcc_neighbor_distances(a, r_max)
    sigma = np.sqrt(B)  # thermal broadening

    G = np.zeros_like(r)
    rho0 = 4 / a**3  # FCC number density

    for d_n, coord_n in shells:
        # Peak amplitude: proportional to coordination / (4πr²ρ₀)
        # Broadened by Gaussian with width sigma that increases with distance
        sigma_n = sigma * np.sqrt(1 + 0.002 * d_n**2)  # slight distance dependence
        amplitude = coord_n / (4 * np.pi * d_n**2 * rho0)
        peak = amplitude * np.exp(-0.5 * ((r - d_n) / sigma_n)**2) / (sigma_n * np.sqrt(2 * np.pi))
        G += peak

    # Subtract baseline (4πrρ₀ term)
    G = scale * G / np.max(np.abs(G) + 1e-12) - 0  # normalize
    # Apply envelope damping for finite Q range
    G *= np.exp(-0.01 * r**2)

    return G


# ═══════════════════════════════════════════════════════════════════
# 3. INVERSE: Least-squares fitting
# ═══════════════════════════════════════════════════════════════════
def residual_func(params, r, G_measured):
    """Residual for least-squares fitting."""
    a, B, scale = params
    if a < 2.0 or a > 6.0 or B < 0.01 or B > 2.0 or scale < 0.1 or scale > 5.0:
        return np.ones_like(r) * 1e6
    G_model = compute_pdf(r, a, B, scale, r_max=R_MAX)
    return G_model - G_measured


def fit_pdf(r, G_measured):
    """
    Fit structural parameters from measured G(r) using least-squares.

    Parameters to fit:
      - a: lattice constant
      - B: Debye-Waller factor
      - scale: overall scale factor
    """
    # Initial guess (slightly off from true values)
    a0 = 3.5
    B0 = 0.4
    scale0 = 0.8

    result = least_squares(
        residual_func,
        x0=[a0, B0, scale0],
        args=(r, G_measured),
        bounds=([2.0, 0.01, 0.1], [6.0, 2.0, 5.0]),
        method="trf",
        max_nfev=2000,
    )

    a_fit, B_fit, scale_fit = result.x
    G_fitted = compute_pdf(r, a_fit, B_fit, scale_fit, r_max=R_MAX)

    return a_fit, B_fit, scale_fit, G_fitted, result


# ═══════════════════════════════════════════════════════════════════
# 4. METRICS
# ═══════════════════════════════════════════════════════════════════
def compute_metrics(r, G_gt, G_fit, a_true, a_fit, B_true, B_fit, scale_true, scale_fit):
    """Compute PSNR, CC, parameter errors."""
    # Normalize
    g_max = np.max(np.abs(G_gt)) + 1e-12
    gt_n = G_gt / g_max
    fi_n = G_fit / g_max

    # PSNR
    mse = np.mean((gt_n - fi_n)**2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-12))

    # CC
    g = gt_n - np.mean(gt_n)
    f = fi_n - np.mean(fi_n)
    cc = np.sum(g * f) / (np.sqrt(np.sum(g**2) * np.sum(f**2)) + 1e-12)

    # Parameter relative errors
    a_err = abs(a_fit - a_true) / a_true * 100
    B_err = abs(B_fit - B_true) / B_true * 100
    scale_err = abs(scale_fit - scale_true) / scale_true * 100

    return {
        "PSNR": float(psnr),
        "CC": float(cc),
        "lattice_constant_true": float(a_true),
        "lattice_constant_fitted": float(a_fit),
        "lattice_constant_error_pct": float(a_err),
        "debye_waller_true": float(B_true),
        "debye_waller_fitted": float(B_fit),
        "debye_waller_error_pct": float(B_err),
        "scale_true": float(scale_true),
        "scale_fitted": float(scale_fit),
        "scale_error_pct": float(scale_err),
        "RMSE": float(np.sqrt(mse)),
    }


# ═══════════════════════════════════════════════════════════════════
# 5. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════
def visualize(r, G_gt, G_noisy, G_fit, metrics):
    """Plot GT vs fitted G(r) with peak identification."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: GT and fitted G(r)
    axes[0, 0].plot(r, G_gt, "b-", linewidth=1.5, label="Ground Truth G(r)")
    axes[0, 0].plot(r, G_fit, "r--", linewidth=1.5, label="Fitted G(r)")
    axes[0, 0].set_xlabel("r (Å)", fontsize=12)
    axes[0, 0].set_ylabel("G(r)", fontsize=12)
    axes[0, 0].set_title("PDF: Ground Truth vs Fit", fontsize=14)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].set_xlim([R_MIN, 15])

    # Panel 2: Noisy data vs fit
    axes[0, 1].plot(r, G_noisy, "gray", alpha=0.6, linewidth=0.8, label="Noisy data")
    axes[0, 1].plot(r, G_fit, "r-", linewidth=1.5, label="Fitted G(r)")
    axes[0, 1].set_xlabel("r (Å)", fontsize=12)
    axes[0, 1].set_ylabel("G(r)", fontsize=12)
    axes[0, 1].set_title("Noisy Data vs Fit", fontsize=14)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].set_xlim([R_MIN, 15])

    # Panel 3: Residual
    residual = G_gt - G_fit
    axes[1, 0].plot(r, residual, "g-", linewidth=1.0)
    axes[1, 0].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[1, 0].set_xlabel("r (Å)", fontsize=12)
    axes[1, 0].set_ylabel("Residual", fontsize=12)
    axes[1, 0].set_title(f"Residual (RMSE = {metrics['RMSE']:.4f})", fontsize=14)
    axes[1, 0].set_xlim([R_MIN, 15])

    # Panel 4: Parameter table
    axes[1, 1].axis("off")
    table_data = [
        ["Parameter", "True", "Fitted", "Error (%)"],
        ["a (Å)", f"{metrics['lattice_constant_true']:.4f}",
         f"{metrics['lattice_constant_fitted']:.4f}",
         f"{metrics['lattice_constant_error_pct']:.2f}%"],
        ["B (Å²)", f"{metrics['debye_waller_true']:.4f}",
         f"{metrics['debye_waller_fitted']:.4f}",
         f"{metrics['debye_waller_error_pct']:.2f}%"],
        ["Scale", f"{metrics['scale_true']:.4f}",
         f"{metrics['scale_fitted']:.4f}",
         f"{metrics['scale_error_pct']:.2f}%"],
        ["", "", "", ""],
        ["PSNR", f"{metrics['PSNR']:.2f} dB", "", ""],
        ["CC", f"{metrics['CC']:.4f}", "", ""],
    ]
    table = axes[1, 1].table(
        cellText=table_data, loc="center", cellLoc="center",
        colWidths=[0.25, 0.25, 0.25, 0.25]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    axes[1, 1].set_title("Fitted Parameters", fontsize=14)

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
    print("diffpy_pdf — Pair Distribution Function Analysis")
    print("=" * 60)

    # 1. r grid
    r = np.arange(R_MIN, R_MAX, DR)

    # 2. Ground truth G(r)
    print("[1/4] Computing ground truth G(r) for FCC Cu ...")
    G_gt = compute_pdf(r, A_TRUE, B_TRUE, SCALE_TRUE, R_MAX)

    # 3. Add noise
    print("[2/4] Adding measurement noise ...")
    noise = NOISE_LEVEL * np.max(np.abs(G_gt)) * np.random.randn(len(r))
    G_noisy = G_gt + noise

    # 4. Fit
    print("[3/4] Fitting structural parameters ...")
    a_fit, B_fit, scale_fit, G_fit, result = fit_pdf(r, G_noisy)
    print(f"  Fitted a     = {a_fit:.4f} Å  (true: {A_TRUE:.4f})")
    print(f"  Fitted B     = {B_fit:.4f} Å² (true: {B_TRUE:.4f})")
    print(f"  Fitted scale = {scale_fit:.4f}   (true: {SCALE_TRUE:.4f})")

    # 5. Metrics
    metrics = compute_metrics(r, G_gt, G_fit, A_TRUE, a_fit, B_TRUE, B_fit, SCALE_TRUE, scale_fit)
    print(f"  PSNR = {metrics['PSNR']:.2f} dB")
    print(f"  CC   = {metrics['CC']:.4f}")

    # 6. Save
    print("[4/4] Saving results ...")
    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), G_gt)
        np.save(os.path.join(d, "recon_output.npy"), G_fit)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    visualize(r, G_gt, G_noisy, G_fit, metrics)

    print("Done ✓")
    return metrics


if __name__ == "__main__":
    main()
