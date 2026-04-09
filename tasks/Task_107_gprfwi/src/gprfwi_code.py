"""
gprfwi - GPR Full-Waveform Inversion
======================================
From ground-penetrating radar (GPR) B-scan data, perform full-waveform
inversion to recover subsurface permittivity distribution.

Physics:
  - Forward: 1D convolutional model per trace
    B-scan = reflectivity convolved with source wavelet + noise
    r(z) = Δε(z) / (2ε(z))  (reflection coefficients from permittivity contrasts)
    trace(t) = wavelet * r(z(t)) + noise
  - Inverse: Deconvolution + impedance inversion (cumulative product of reflectivity)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim


def ricker_wavelet(points, a):
    """Ricker wavelet (Mexican hat). Equivalent to scipy.signal.ricker."""
    A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
    wsq = a**2
    vec = np.arange(0, points) - (points - 1.0) / 2
    tsq = vec**2
    mod = (1 - tsq / wsq)
    gauss = np.exp(-tsq / (2 * wsq))
    return A * mod * gauss

# ── paths ──────────────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR  = "/data/yjh/website_assets/Task_107_gprfwi"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── parameters ─────────────────────────────────────────────────────
NZ          = 200       # depth samples
NX          = 80        # number of traces (lateral positions)
NOISE_LEVEL = 0.01      # 1% additive Gaussian noise on B-scan
WAVELET_PTS = 31        # Ricker wavelet width
WAVELET_A   = 4         # Ricker wavelet parameter (controls frequency)
SEED        = 42
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════
# 1. GROUND TRUTH: 2D subsurface permittivity model
# ═══════════════════════════════════════════════════════════════════
def create_gt_permittivity(nz, nx):
    """
    Create a 2D permittivity model with horizontal layers and anomalies.
    Shape: (nz, nx).  Values represent relative permittivity ε_r.
    """
    eps = np.ones((nz, nx)) * 4.0       # background ε_r = 4 (dry sand)

    # Layer 1: top soil (ε_r = 6) from z=30 to z=60
    eps[30:60, :] = 6.0

    # Layer 2: wet clay (ε_r = 15) from z=80 to z=120
    eps[80:120, :] = 15.0

    # Layer 3: bedrock (ε_r = 8) from z=150 onward
    eps[150:, :] = 8.0

    # Anomaly 1: buried pipe (ε_r = 1, air) — circle at (z=45, x=25), r=5
    zz, xx = np.ogrid[:nz, :nx]
    mask1 = (zz - 45)**2 + (xx - 25)**2 < 5**2
    eps[mask1] = 1.0

    # Anomaly 2: water pocket (ε_r = 80) — ellipse at (z=100, x=55)
    mask2 = ((zz - 100) / 6)**2 + ((xx - 55) / 8)**2 < 1
    eps[mask2] = 40.0

    # Anomaly 3: metallic object (ε_r = 30) — small rect at (z=140, x=40)
    eps[137:143, 37:43] = 30.0

    # Smooth slightly to avoid unrealistically sharp transitions
    eps = gaussian_filter(eps, sigma=1.0)
    return eps


# ═══════════════════════════════════════════════════════════════════
# 2. FORWARD MODEL
# ═══════════════════════════════════════════════════════════════════
def compute_reflection_coefficients(eps_profile):
    """
    From a 1D permittivity profile ε(z), compute reflection coefficients.
    r(z) = (sqrt(ε(z+1)) - sqrt(ε(z))) / (sqrt(ε(z+1)) + sqrt(ε(z)))
    """
    sqrt_eps = np.sqrt(eps_profile)
    r = np.zeros_like(eps_profile)
    r[:-1] = (sqrt_eps[1:] - sqrt_eps[:-1]) / (sqrt_eps[1:] + sqrt_eps[:-1] + 1e-12)
    return r


def forward_gpr(eps_model, wavelet):
    """
    Simplified GPR forward model: convolve each column's reflectivity with wavelet.
    Returns B-scan (nz_out, nx) and clean reflectivity (nz, nx).
    """
    nz, nx = eps_model.shape
    # Compute reflectivity for each trace
    reflectivity = np.zeros_like(eps_model)
    for ix in range(nx):
        reflectivity[:, ix] = compute_reflection_coefficients(eps_model[:, ix])

    # Convolve each trace with wavelet
    bscan = np.zeros_like(reflectivity)
    half_w = len(wavelet) // 2
    for ix in range(nx):
        conv = fftconvolve(reflectivity[:, ix], wavelet, mode='same')
        bscan[:, ix] = conv

    return bscan, reflectivity


def add_noise(data, noise_level):
    """Add Gaussian noise scaled to signal amplitude."""
    amp = np.max(np.abs(data)) + 1e-12
    noise = np.random.randn(*data.shape) * amp * noise_level
    return data + noise


# ═══════════════════════════════════════════════════════════════════
# 3. INVERSE: Deconvolution + Impedance Inversion
# ═══════════════════════════════════════════════════════════════════
def build_convolution_matrix(wavelet, n):
    """
    Build the convolution matrix H such that H @ r ≈ fftconvolve(r, wavelet, 'same').
    """
    w_len = len(wavelet)
    half_w = w_len // 2
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(w_len):
            col = i - half_w + j
            if 0 <= col < n:
                H[i, col] = wavelet[j]
    return H


def tikhonov_deconvolution(bscan_trace, H, HtH, lam_reg=0.003):
    """
    Tikhonov-regularized least-squares deconvolution:
    r_est = argmin ||H @ r - b||^2 + λ||r||^2
    Solved via: (H^T H + λI) r = H^T b
    """
    n = len(bscan_trace)
    I = np.eye(n)
    A = HtH + lam_reg * I
    r = np.linalg.solve(A, H.T @ bscan_trace)
    return r


def wiener_deconvolution(bscan_trace, wavelet, noise_power_fraction=0.01):
    """
    Wiener deconvolution: recover reflectivity from convolved trace.
    R(f) = B(f) * W*(f) / (|W(f)|^2 + λ)
    """
    n = len(bscan_trace)
    B = np.fft.fft(bscan_trace, n=n)
    W = np.fft.fft(wavelet, n=n)
    power = np.abs(W)**2
    lam = noise_power_fraction * np.max(power)
    R = B * np.conj(W) / (power + lam)
    r = np.real(np.fft.ifft(R))
    return r


def impedance_inversion(reflectivity_profile, eps_surface=4.0):
    """
    From reflectivity r(z), recover permittivity via impedance inversion.
    Z(z+1) = Z(z) * (1 + r(z)) / (1 - r(z))
    ε(z) = Z(z)^2  (assuming μ=1)
    Actually Z ∝ sqrt(ε), so ε ∝ Z^2.
    We use: sqrt(ε(z+1)) = sqrt(ε(z)) * (1 + r(z)) / (1 - r(z))
    """
    n = len(reflectivity_profile)
    sqrt_eps = np.zeros(n)
    sqrt_eps[0] = np.sqrt(eps_surface)

    for i in range(n - 1):
        r = np.clip(reflectivity_profile[i], -0.95, 0.95)
        sqrt_eps[i + 1] = sqrt_eps[i] * (1 + r) / (1 - r)

    # Ensure positivity
    sqrt_eps = np.clip(sqrt_eps, 0.1, 100.0)
    eps = sqrt_eps ** 2
    return eps


def inverse_gpr(bscan_noisy, wavelet, eps_surface=4.0):
    """
    Full inverse pipeline: Tikhonov-regularized deconvolution + impedance inversion.
    Uses least-squares with Tikhonov regularization instead of Wiener filter
    for more accurate reflectivity recovery.
    """
    nz, nx = bscan_noisy.shape
    eps_recon = np.zeros((nz, nx))

    # Pre-compute convolution matrix and H^T H (shared across traces)
    H = build_convolution_matrix(wavelet, nz)
    HtH = H.T @ H

    for ix in range(nx):
        # Tikhonov-regularized deconvolution (more accurate than Wiener)
        r_est = tikhonov_deconvolution(bscan_noisy[:, ix], H, HtH,
                                        lam_reg=0.003)
        # Impedance inversion
        eps_recon[:, ix] = impedance_inversion(r_est, eps_surface)

    # Smoothing to reduce noise artifacts (sigma=3.0 for better denoising)
    eps_recon = gaussian_filter(eps_recon, sigma=3.0)
    return eps_recon


# ═══════════════════════════════════════════════════════════════════
# 4. METRICS
# ═══════════════════════════════════════════════════════════════════
def compute_psnr(gt, recon):
    """PSNR in dB."""
    mse = np.mean((gt - recon)**2)
    if mse < 1e-15:
        return 100.0
    data_range = np.max(gt) - np.min(gt)
    return 10 * np.log10(data_range**2 / mse)


def compute_cc(gt, recon):
    """Pearson correlation coefficient."""
    g = gt.ravel() - np.mean(gt)
    r = recon.ravel() - np.mean(recon)
    denom = np.sqrt(np.sum(g**2) * np.sum(r**2))
    if denom < 1e-15:
        return 0.0
    return float(np.sum(g * r) / denom)


def compute_ssim(gt, recon):
    """SSIM for 2D images."""
    data_range = np.max(gt) - np.min(gt)
    if data_range < 1e-10:
        data_range = 1.0
    return float(ssim(gt, recon, data_range=data_range))


# ═══════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("GPR Full-Waveform Inversion")
    print("=" * 60)

    # Create GT permittivity model
    eps_gt = create_gt_permittivity(NZ, NX)
    print(f"GT permittivity shape: {eps_gt.shape}, range: [{eps_gt.min():.2f}, {eps_gt.max():.2f}]")

    # Create source wavelet (Ricker)
    wavelet = ricker_wavelet(WAVELET_PTS, WAVELET_A)

    # Forward model: generate B-scan
    bscan_clean, reflectivity_gt = forward_gpr(eps_gt, wavelet)
    bscan_noisy = add_noise(bscan_clean, NOISE_LEVEL)
    print(f"B-scan shape: {bscan_noisy.shape}")

    # Inverse: recover permittivity
    eps_recon = inverse_gpr(bscan_noisy, wavelet, eps_surface=4.0)
    print(f"Reconstructed permittivity shape: {eps_recon.shape}")

    # Metrics
    psnr_val = compute_psnr(eps_gt, eps_recon)
    ssim_val = compute_ssim(eps_gt, eps_recon)
    cc_val   = compute_cc(eps_gt, eps_recon)
    print(f"PSNR: {psnr_val:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")
    print(f"CC:   {cc_val:.4f}")

    # Save metrics
    metrics = {"PSNR": float(psnr_val), "SSIM": float(ssim_val), "CC": float(cc_val)}
    for path in [RESULTS_DIR, ASSETS_DIR]:
        with open(os.path.join(path, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    # Save numpy outputs
    for path in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(path, "gt_output.npy"), eps_gt)
        np.save(os.path.join(path, "recon_output.npy"), eps_recon)

    # ── Visualization ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # B-scan
    ax = axes[0, 0]
    im = ax.imshow(bscan_noisy, aspect='auto', cmap='seismic',
                   vmin=-np.max(np.abs(bscan_noisy)), vmax=np.max(np.abs(bscan_noisy)))
    ax.set_title("GPR B-scan (noisy)")
    ax.set_xlabel("Trace index")
    ax.set_ylabel("Depth sample")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # GT permittivity
    ax = axes[0, 1]
    im = ax.imshow(eps_gt, aspect='auto', cmap='viridis')
    ax.set_title("GT Permittivity εᵣ")
    ax.set_xlabel("Lateral position")
    ax.set_ylabel("Depth sample")
    plt.colorbar(im, ax=ax, shrink=0.8, label="εᵣ")

    # Reconstructed permittivity
    ax = axes[1, 0]
    im = ax.imshow(eps_recon, aspect='auto', cmap='viridis',
                   vmin=eps_gt.min(), vmax=eps_gt.max())
    ax.set_title(f"Reconstructed εᵣ (PSNR={psnr_val:.1f}dB)")
    ax.set_xlabel("Lateral position")
    ax.set_ylabel("Depth sample")
    plt.colorbar(im, ax=ax, shrink=0.8, label="εᵣ")

    # Error map
    ax = axes[1, 1]
    error = np.abs(eps_gt - eps_recon)
    im = ax.imshow(error, aspect='auto', cmap='hot')
    ax.set_title(f"Absolute Error (SSIM={ssim_val:.3f}, CC={cc_val:.3f})")
    ax.set_xlabel("Lateral position")
    ax.set_ylabel("Depth sample")
    plt.colorbar(im, ax=ax, shrink=0.8, label="|error|")

    plt.suptitle("GPR Full-Waveform Inversion", fontsize=14, fontweight='bold')
    plt.tight_layout()

    for path in [RESULTS_DIR, ASSETS_DIR]:
        fig.savefig(os.path.join(path, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(path, "vis_result.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nResults saved to {RESULTS_DIR}")
    print(f"Assets saved to {ASSETS_DIR}")
    print("DONE")


if __name__ == "__main__":
    main()
