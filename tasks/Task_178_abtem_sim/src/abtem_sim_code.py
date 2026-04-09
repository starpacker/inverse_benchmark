#!/usr/bin/env python
"""
Task 178: abtem_sim — HRTEM inverse parameter estimation.

Forward model (phase-object approximation + CTF):
  phase(x,y) = sigma_e * V(x,y) * t
  psi_exit   = exp(i * phase)
  Image      = |IFFT{ FFT{psi_exit} * CTF(k) }|^2

Inverse: estimate defocus and thickness from noisy image via grid-search + refinement.
"""

import json
import os
import numpy as np
from scipy.optimize import minimize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def relativistic_wavelength_A(voltage_kv):
    """Relativistic de Broglie wavelength in Angstroms."""
    V = voltage_kv * 1e3  # Volts
    return 12.2643 / np.sqrt(V + 0.97845e-6 * V**2)


def interaction_param(voltage_kv):
    """
    Interaction parameter sigma_e in rad/(V*Å).
    Formula: sigma = (2*pi / (lambda*V)) * (mc^2 + eV) / (2*mc^2 + eV)
    where lambda in Å, V in eV, mc^2 = 510998.95 eV.
    """
    lam = relativistic_wavelength_A(voltage_kv)  # Å
    V = voltage_kv * 1e3  # eV
    mc2 = 510998.95  # eV
    sigma = 2 * np.pi / (lam * V) * (mc2 + V) / (2 * mc2 + V)
    return sigma  # rad / (V·Å)


def make_si110_potential(nx=256, ny=256, pixel_size=0.05):
    """
    Create 2D projected potential for Si [110] zone axis.
    Returns V(x,y) in Volts.
    """
    a, b = 3.84, 5.43  # Si [110] unit cell (Å)
    positions = [(0.0, 0.0), (0.5, 0.5), (0.25, 0.25), (0.75, 0.75)]

    Lx = nx * pixel_size
    Ly = ny * pixel_size
    x = np.arange(nx) * pixel_size
    y = np.arange(ny) * pixel_size
    X, Y = np.meshgrid(x, y, indexing='xy')

    V = np.zeros((ny, nx), dtype=np.float64)
    sigma_atom = 0.30  # Gaussian width (Å)
    V0 = 15.0          # peak potential (V)

    for ix in range(int(np.ceil(Lx / a)) + 1):
        for iy in range(int(np.ceil(Ly / b)) + 1):
            for fx, fy in positions:
                ax = ix * a + fx * a
                ay = iy * b + fy * b
                V += V0 * np.exp(-((X - ax)**2 + (Y - ay)**2) / (2 * sigma_atom**2))
    return V


def ctf(k, lam, defocus_nm, cs_mm):
    """
    Contrast transfer function H(k).
    defocus_nm: negative = underfocus (standard for HRTEM).
    """
    df_A = defocus_nm * 10.0
    Cs_A = cs_mm * 1e7

    chi = np.pi * lam * df_A * k**2 - 0.5 * np.pi * Cs_A * lam**3 * k**4

    # Spatial coherence envelope
    alpha = 0.5e-3
    E_s = np.exp(-0.5 * (np.pi * alpha)**2 * (df_A * k + Cs_A * lam**2 * k**3)**2)
    # Temporal coherence envelope
    delta_f = 30.0  # Å
    E_t = np.exp(-0.5 * (np.pi * lam * delta_f)**2 * k**4)

    return np.exp(-1j * chi) * E_s * E_t


def freq_grid(nx, ny, pixel_size):
    """2D spatial frequency magnitude |k| in 1/Å."""
    kx = np.fft.fftfreq(nx, d=pixel_size)
    ky = np.fft.fftfreq(ny, d=pixel_size)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    return np.sqrt(KX**2 + KY**2)


def forward(V_pot, thickness_nm, defocus_nm, voltage_kv=200.0, cs_mm=1.0,
            pixel_size=0.05, noise_level=0.0, rng=None):
    """
    HRTEM forward model.
    V_pot: potential in Volts, thickness in nm, defocus in nm.
    """
    ny, nx = V_pot.shape
    lam = relativistic_wavelength_A(voltage_kv)
    sig = interaction_param(voltage_kv)
    t_A = thickness_nm * 10.0  # nm → Å

    phase = sig * V_pot * t_A
    psi_exit = np.exp(1j * phase)

    K = freq_grid(nx, ny, pixel_size)
    H = ctf(K, lam, defocus_nm, cs_mm)

    psi_img = np.fft.ifft2(np.fft.fft2(psi_exit) * H)
    img = np.abs(psi_img)**2

    if noise_level > 0 and rng is not None:
        img = img + rng.normal(0, noise_level * img.mean(), img.shape)
        img = np.maximum(img, 0.0)

    return img


def normalize(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-15)


def main():
    os.makedirs("results", exist_ok=True)

    NX, NY = 256, 256
    PIXEL = 0.05
    VKV = 200.0
    CS = 1.0
    TRUE_DF = -50.0
    TRUE_T = 5.0
    NOISE = 0.02

    print("=" * 60)
    print("Task 178: abtem_sim — HRTEM inverse parameter estimation")
    print("=" * 60)

    lam = relativistic_wavelength_A(VKV)
    sig = interaction_param(VKV)
    print(f"  λ = {lam:.5f} Å,  σ_e = {sig:.6f} rad/(V·Å)")

    # 1. Build potential
    print("\n[1/5] Building Si [110] projected potential ...")
    V_pot = make_si110_potential(NX, NY, PIXEL)
    print(f"      V range: [{V_pot.min():.2f}, {V_pot.max():.2f}] V")
    print(f"      Max phase: {sig * V_pot.max() * TRUE_T * 10:.3f} rad")

    # 2. Simulate images
    print("[2/5] Simulating GT and noisy images ...")
    gt_img = forward(V_pot, TRUE_T, TRUE_DF, VKV, CS, PIXEL)
    rng = np.random.default_rng(42)
    noisy_img = forward(V_pot, TRUE_T, TRUE_DF, VKV, CS, PIXEL,
                        noise_level=NOISE, rng=rng)

    contrast = (gt_img.max() - gt_img.min()) / gt_img.mean()
    print(f"      GT range: [{gt_img.min():.6f}, {gt_img.max():.6f}], contrast={contrast:.4f}")

    # 3. Coarse grid search
    print("[3/5] Coarse grid search ...")
    df_grid = np.linspace(-100, 0, 41)
    t_grid = np.linspace(1, 10, 37)

    best_cost, best_df, best_t = np.inf, df_grid[0], t_grid[0]
    for df in df_grid:
        for t in t_grid:
            sim = forward(V_pot, t, df, VKV, CS, PIXEL)
            c = np.mean((sim - noisy_img)**2)
            if c < best_cost:
                best_cost, best_df, best_t = c, df, t

    print(f"      Best: df={best_df:.1f}, t={best_t:.2f}  (MSE={best_cost:.4e})")

    # 4. Refine
    print("[4/5] Refining ...")
    def cost(p):
        d, t = p
        if t < 0.1 or t > 20:
            return 1e10
        return np.mean((forward(V_pot, t, d, VKV, CS, PIXEL) - noisy_img)**2)

    res = minimize(cost, [best_df, best_t], method='Nelder-Mead',
                   options={'xatol': 0.05, 'fatol': 1e-14, 'maxiter': 800, 'adaptive': True})
    est_df, est_t = res.x
    print(f"      Refined: df={est_df:.3f}, t={est_t:.4f}")

    recon_img = forward(V_pot, est_t, est_df, VKV, CS, PIXEL)

    # 5. Evaluate
    print("[5/5] Evaluating ...")
    gt_n = normalize(gt_img)
    recon_n = normalize(recon_img)
    noisy_n = normalize(noisy_img)

    psnr_r = peak_signal_noise_ratio(gt_n, recon_n, data_range=1.0)
    ssim_r = structural_similarity(gt_n, recon_n, data_range=1.0)
    cc_r = float(np.corrcoef(gt_n.ravel(), recon_n.ravel())[0, 1])

    psnr_n = peak_signal_noise_ratio(gt_n, noisy_n, data_range=1.0)
    ssim_n = structural_similarity(gt_n, noisy_n, data_range=1.0)

    re_df = abs(est_df - TRUE_DF) / abs(TRUE_DF)
    re_t = abs(est_t - TRUE_T) / abs(TRUE_T)

    print(f"\n{'─'*55}")
    print(f"  TRUE : df={TRUE_DF:.1f} nm, t={TRUE_T:.2f} nm")
    print(f"  EST  : df={est_df:.3f} nm, t={est_t:.4f} nm")
    print(f"  RE   : df={re_df:.6f}, t={re_t:.6f}")
    print(f"  Recon: PSNR={psnr_r:.2f}, SSIM={ssim_r:.4f}, CC={cc_r:.4f}")
    print(f"  Noisy: PSNR={psnr_n:.2f}, SSIM={ssim_n:.4f}")
    print(f"{'─'*55}\n")

    # Save metrics
    metrics = {
        "task_id": 178,
        "task_name": "abtem_sim",
        "inverse_problem": "HRTEM inverse parameter estimation (defocus + thickness)",
        "true_defocus_nm": TRUE_DF,
        "true_thickness_nm": TRUE_T,
        "estimated_defocus_nm": round(float(est_df), 3),
        "estimated_thickness_nm": round(float(est_t), 4),
        "defocus_relative_error": round(float(re_df), 6),
        "thickness_relative_error": round(float(re_t), 6),
        "reconstruction_PSNR_dB": round(float(psnr_r), 2),
        "reconstruction_SSIM": round(float(ssim_r), 4),
        "reconstruction_CC": round(float(cc_r), 4),
        "noisy_PSNR_dB": round(float(psnr_n), 2),
        "noisy_SSIM": round(float(ssim_n), 4),
    }
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved results/metrics.json")

    np.save("results/ground_truth.npy", gt_img)
    np.save("results/noisy_observation.npy", noisy_img)
    np.save("results/reconstruction.npy", recon_img)
    np.save("results/projected_potential.npy", V_pot)
    print("Saved .npy arrays")

    # Visualisation
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    im0 = axes[0, 0].imshow(gt_n, cmap="gray", origin="lower")
    axes[0, 0].set_title("(a) GT noiseless HRTEM image", fontsize=11)
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(noisy_n, cmap="gray", origin="lower")
    axes[0, 1].set_title("(b) Noisy observation", fontsize=11)
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(recon_n, cmap="gray", origin="lower")
    axes[1, 0].set_title(f"(c) Best-fit image\ndf={est_df:.2f} nm, t={est_t:.3f} nm", fontsize=11)
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    err = np.abs(gt_n - recon_n)
    im3 = axes[1, 1].imshow(err, cmap="hot", origin="lower")
    axes[1, 1].set_title(f"(d) |GT − Recon| error\nPSNR={psnr_r:.1f} dB, SSIM={ssim_r:.3f}", fontsize=11)
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Task 178: abtem_sim — HRTEM Inverse Parameter Estimation\n"
        f"True: df={TRUE_DF} nm, t={TRUE_T} nm  |  Est: df={est_df:.2f} nm, t={est_t:.3f} nm\n"
        f"Defocus RE={re_df:.4f}, Thickness RE={re_t:.4f}",
        fontsize=12, fontweight="bold", y=1.01)

    for ax in axes.ravel():
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")

    plt.tight_layout()
    fig.savefig("results/reconstruction_result.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved results/reconstruction_result.png\n\nDone.")


if __name__ == "__main__":
    main()
