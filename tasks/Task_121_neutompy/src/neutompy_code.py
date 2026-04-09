"""
neutompy — Neutron Tomography Reconstruction
=============================================
From neutron transmission radiographs at multiple angles, reconstruct the 2D
attenuation coefficient distribution using Filtered Backprojection (FBP).

Physics:
  - Forward: Beer-Lambert law: I(θ,s) = I_0 × exp(-∫ μ(x,y) dl)
    → sinogram: p(θ,s) = -ln(I/I_0) = Radon transform of μ(x,y)
  - Inverse: Filtered Backprojection with Ram-Lak filter
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from skimage.transform import radon, iradon

# ── paths ──────────────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR  = "/data/yjh/website_assets/Task_112_neutompy"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── parameters ─────────────────────────────────────────────────────
N            = 256       # image size
N_ANGLES     = 180       # number of projection angles
NOISE_LEVEL  = 0.02      # Poisson-like noise level
I0           = 1e4       # incident neutron flux (counts)
SEED         = 42
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════
# 1. GROUND TRUTH: 2D attenuation coefficient phantom
# ═══════════════════════════════════════════════════════════════════
def create_phantom(n):
    """
    Create a 2D cross-section phantom mimicking metal cylinders embedded
    in a matrix material (typical neutron tomography sample).

    Attenuation coefficients per pixel (dimensionless).
    Scaled so that max line integral through the sample ≈ 3-5,
    giving meaningful Beer-Lambert transmission.
    
    Physical analogy (thermal neutrons, ~10cm sample on 256 pixels):
      pixel_size ≈ 0.04 cm → μ_pixel = μ_physical × pixel_size
      - Air/void:    0
      - Aluminum:    μ ≈ 0.1 cm⁻¹  → 0.004/pixel
      - Steel:       μ ≈ 1.0 cm⁻¹  → 0.04/pixel
      - Water:       μ ≈ 3.5 cm⁻¹  → 0.14/pixel
      - Lead:        μ ≈ 0.4 cm⁻¹  → 0.016/pixel
    """
    phantom = np.zeros((n, n), dtype=np.float64)
    yy, xx = np.ogrid[:n, :n]
    c = n // 2

    # Scale factor: physical_mu * pixel_size
    s = 0.04  # pixel size in cm

    # Outer aluminum casing (cylinder)
    r_outer = np.sqrt((yy - c)**2 + (xx - c)**2)
    phantom[r_outer < 0.42 * n] = 0.1 * s   # aluminum matrix

    # Steel cylinder 1 (off-center)
    r1 = np.sqrt((yy - c + 30)**2 + (xx - c - 40)**2)
    phantom[r1 < 25] = 1.0 * s

    # Steel cylinder 2
    r2 = np.sqrt((yy - c - 35)**2 + (xx - c + 25)**2)
    phantom[r2 < 20] = 1.0 * s

    # Water-filled cavity (high attenuation for neutrons)
    r3 = np.sqrt((yy - c + 10)**2 + (xx - c + 10)**2)
    phantom[r3 < 15] = 3.5 * s

    # Lead inclusion (moderate attenuation)
    r4 = np.sqrt((yy - c - 20)**2 + (xx - c - 30)**2)
    phantom[r4 < 12] = 0.4 * s

    # Small void (crack/pore)
    r5 = np.sqrt((yy - c + 40)**2 + (xx - c + 50)**2)
    phantom[r5 < 8] = 0.0

    # Another small steel rod
    r6 = np.sqrt((yy - c - 50)**2 + (xx - c - 10)**2)
    phantom[r6 < 10] = 0.8 * s

    # Annular structure
    r7 = np.sqrt((yy - c + 45)**2 + (xx - c - 15)**2)
    phantom[(r7 > 12) & (r7 < 18)] = 1.2 * s

    return phantom


# ═══════════════════════════════════════════════════════════════════
# 2. FORWARD MODEL: Beer-Lambert + Radon
# ═══════════════════════════════════════════════════════════════════
def forward_neutron_tomo(phantom, n_angles, noise_level):
    """
    Simulate neutron transmission radiography.

    Physical process:
    1. Neutron beam traverses the sample: line integrals of μ(x,y)
    2. Beer-Lambert law: I = I_0 exp(-∫μ dl) → transmission image
    3. Poisson counting noise corrupts transmission
    4. Reconstruction works on sinogram = -ln(I/I_0) ≈ Radon transform + noise

    For the simulation, we:
    1. Compute ideal sinogram via Radon transform
    2. Add realistic noise (Gaussian approximation of Poisson in log domain)
    """
    angles = np.linspace(0, 180, n_angles, endpoint=False)

    # Radon transform → sinogram (ideal line integrals of μ)
    sinogram_ideal = radon(phantom, theta=angles, circle=False)

    # Beer-Lambert noise model:
    # I = I0 * exp(-sinogram) → Poisson noise → -ln(I_noisy / I0)
    transmitted = I0 * np.exp(-sinogram_ideal)
    transmitted_noisy = np.random.poisson(
        np.maximum(transmitted, 1).astype(np.float64)
    ).astype(np.float64)
    # Add Gaussian readout noise (small)
    transmitted_noisy += np.random.normal(0, 2.0, transmitted_noisy.shape)
    transmitted_noisy = np.maximum(transmitted_noisy, 1.0)  # avoid log(0)
    sinogram_noisy = -np.log(transmitted_noisy / I0)

    return sinogram_noisy, sinogram_ideal, angles


# ═══════════════════════════════════════════════════════════════════
# 3. INVERSE: Filtered Backprojection
# ═══════════════════════════════════════════════════════════════════
def fbp_reconstruct(sinogram, angles):
    """
    Filtered Backprojection (FBP) with Ram-Lak filter.
    
    Note: We pass the sinogram directly (already as line-integral data).
    skimage.transform.iradon expects sinogram = Radon transform output.
    """
    recon = iradon(sinogram, theta=angles, filter_name="ramp", circle=False)
    recon = np.maximum(recon, 0)  # physical: attenuation >= 0
    return recon


# ═══════════════════════════════════════════════════════════════════
# 4. METRICS
# ═══════════════════════════════════════════════════════════════════
def compute_metrics(gt, recon):
    """Compute PSNR, SSIM, RMSE."""
    # Crop to same size (iradon may produce slightly different shape)
    min_h = min(gt.shape[0], recon.shape[0])
    min_w = min(gt.shape[1], recon.shape[1])
    gt_c = gt[:min_h, :min_w]
    re_c = recon[:min_h, :min_w]

    # RMSE
    rmse = np.sqrt(np.mean((gt_c - re_c)**2))

    # Data range
    data_range = gt_c.max() - gt_c.min()
    if data_range < 1e-10:
        data_range = 1.0

    # PSNR
    mse = np.mean((gt_c - re_c)**2)
    psnr = 10 * np.log10(data_range**2 / (mse + 1e-12))

    # SSIM
    ssim_val = ssim(gt_c, re_c, data_range=data_range)

    # CC (Pearson correlation)
    g = gt_c.flatten() - gt_c.mean()
    r = re_c.flatten() - re_c.mean()
    cc = np.sum(g * r) / (np.sqrt(np.sum(g**2) * np.sum(r**2)) + 1e-12)

    return {
        "PSNR": float(psnr),
        "SSIM": float(ssim_val),
        "RMSE": float(rmse),
        "CC": float(cc),
    }


# ═══════════════════════════════════════════════════════════════════
# 5. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════
def visualize(gt, sinogram, recon, metrics):
    """Plot sinogram, GT, FBP reconstruction, error map."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    gt_n = (gt - gt.min()) / (gt.max() - gt.min() + 1e-12)
    re_n = (recon - recon.min()) / (recon.max() - recon.min() + 1e-12)

    im0 = axes[0, 0].imshow(sinogram.T, cmap="gray", aspect="auto",
                             extent=[0, 180, -sinogram.shape[0]//2, sinogram.shape[0]//2])
    axes[0, 0].set_title("Sinogram (neutron transmission)", fontsize=14)
    axes[0, 0].set_xlabel("Angle (degrees)")
    axes[0, 0].set_ylabel("Detector position")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(gt_n, cmap="inferno")
    axes[0, 1].set_title("Ground Truth (μ distribution)", fontsize=14)
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(re_n, cmap="inferno")
    axes[1, 0].set_title(
        f"FBP Reconstruction\nPSNR={metrics['PSNR']:.2f} dB, "
        f"SSIM={metrics['SSIM']:.4f}",
        fontsize=12,
    )
    axes[1, 0].axis("off")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    error = np.abs(gt_n - re_n)
    im3 = axes[1, 1].imshow(error, cmap="magma")
    axes[1, 1].set_title(f"Absolute Error (RMSE={metrics['RMSE']:.4f} cm⁻¹)", fontsize=12)
    axes[1, 1].axis("off")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

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
    print("neutompy — Neutron Tomography Reconstruction")
    print("=" * 60)

    # 1. Create ground truth
    print("[1/4] Creating phantom (attenuation map) ...")
    gt = create_phantom(N)

    # 2. Forward model
    print(f"[2/4] Simulating neutron transmission ({N_ANGLES} angles) ...")
    sinogram, sinogram_ideal, angles = forward_neutron_tomo(gt, N_ANGLES, NOISE_LEVEL)

    # 3. FBP reconstruction
    print("[3/4] Running Filtered Backprojection ...")
    recon = fbp_reconstruct(sinogram, angles)

    # 4. Metrics
    metrics = compute_metrics(gt, recon)
    print(f"  PSNR = {metrics['PSNR']:.2f} dB")
    print(f"  SSIM = {metrics['SSIM']:.4f}")
    print(f"  RMSE = {metrics['RMSE']:.4f}")
    print(f"  CC   = {metrics['CC']:.4f}")

    # 5. Save
    print("[4/4] Saving results ...")
    for d in [RESULTS_DIR, ASSETS_DIR]:
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "gt_output.npy"), gt)
        np.save(os.path.join(d, "recon_output.npy"), recon)
        np.save(os.path.join(d, "ground_truth.npy"), gt)
        np.save(os.path.join(d, "reconstruction.npy"), recon)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    visualize(gt, sinogram, recon, metrics)

    print("Done ✓")
    return metrics


if __name__ == "__main__":
    main()
