#!/usr/bin/env python3
"""
21cm Tomography — Inverse Problem

Forward model: y(ν,θ) = T_21(ν,θ) + T_fg(ν,θ) + n(ν,θ)
    where:
    - T_21 is the cosmological 21cm brightness temperature (~10-30 mK)
    - T_fg is the astrophysical foreground (synchrotron + free-free)
      modeled as smooth power-law in frequency, ~100x stronger
    - n is thermal noise

Inverse problem: Given total observed brightness temperature y(ν,θ),
    separate and recover the faint 21cm signal T_21 from the dominant
    foreground contamination T_fg.

Methods:
    1. Polynomial Fitting: Fit and subtract a low-order polynomial in
       log-frequency space at each angular pixel
    2. PCA-based Foreground Removal: Use SVD to identify and remove the
       spectrally smooth foreground modes

Metrics: PSNR, Correlation Coefficient (CC) of recovered 21cm signal vs GT.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

# ============================================================
# Configuration
# ============================================================
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# Frequency grid: 100–200 MHz (redshifted 21cm line, z ~ 6–13)
N_FREQ = 128
FREQ_MIN = 100.0   # MHz
FREQ_MAX = 200.0   # MHz
FREQUENCIES = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQ)
FREQ_REF = 150.0   # Reference frequency (MHz)

# Angular grid
N_ANGLE = 64

# Signal parameters
T21_RMS = 20.0       # mK — typical 21cm brightness temperature fluctuation
T_FG_AMP = 5.0       # K  — foreground amplitude at reference frequency
BETA_MEAN = 2.5      # Mean spectral index for synchrotron foreground
BETA_STD = 0.05      # Spatial variation in spectral index
NOISE_RMS = 2.0      # mK — thermal noise per pixel

# Method parameters
POLY_ORDER = 4        # Polynomial order for log-frequency fitting
N_PCA_COMPONENTS = 3  # Number of SVD/PCA modes to remove


# ============================================================
# Simulation Functions
# ============================================================

def generate_21cm_signal(n_freq, n_angle, rms=T21_RMS):
    """
    Generate a simulated 21cm brightness temperature field.
    Correlated random field with spectral and angular structure.
    """
    raw = np.random.randn(n_freq, n_angle)

    # Spectral correlation kernel
    fk = max(5, n_freq // 16)
    freq_kernel = np.exp(-0.5 * np.linspace(-2.5, 2.5, fk)**2)
    freq_kernel /= freq_kernel.sum()

    # Angular correlation kernel
    ak = max(5, n_angle // 8)
    ang_kernel = np.exp(-0.5 * np.linspace(-2.5, 2.5, ak)**2)
    ang_kernel /= ang_kernel.sum()

    T21 = np.zeros_like(raw)
    for j in range(n_angle):
        T21[:, j] = np.convolve(raw[:, j], freq_kernel, mode='same')
    for i in range(n_freq):
        T21[i, :] = np.convolve(T21[i, :], ang_kernel, mode='same')

    T21 -= T21.mean()
    T21 = T21 / np.std(T21) * rms
    return T21


def generate_foreground(n_freq, n_angle, frequencies, T_amp=T_FG_AMP,
                        beta_mean=BETA_MEAN, beta_std=BETA_STD, freq_ref=FREQ_REF):
    """
    Generate astrophysical foreground (synchrotron power-law).
    T_fg(ν,θ) = A(θ) * (ν/ν_ref)^{-β(θ)}
    """
    angles_norm = np.linspace(0, 2 * np.pi, n_angle, endpoint=False)
    beta_spatial = beta_mean + beta_std * np.sin(angles_norm * 2)
    amp_spatial = T_amp * (1.0 + 0.2 * np.cos(angles_norm * 3))

    freq_ratio = frequencies[:, np.newaxis] / freq_ref
    T_fg = amp_spatial[np.newaxis, :] * freq_ratio ** (-beta_spatial[np.newaxis, :])
    return T_fg  # K


def forward_model(T21, T_fg, noise_rms=NOISE_RMS):
    """y = T_21 + T_fg*1000 + noise  (all in mK)"""
    noise = noise_rms * np.random.randn(*T21.shape)
    observation = T21 + T_fg * 1000.0 + noise
    return observation, noise


# ============================================================
# Inverse Methods
# ============================================================

def polynomial_foreground_removal(observation, frequencies, poly_order=POLY_ORDER):
    """
    Fit & subtract polynomial in log(ν) space per angular pixel.
    Power-law foregrounds ∝ ν^{-β} are nearly polynomial in log(ν).
    """
    n_freq, n_angle = observation.shape
    residual = np.zeros_like(observation)
    fg_estimate = np.zeros_like(observation)
    log_freq = np.log(frequencies / FREQ_REF)

    for j in range(n_angle):
        coeffs = np.polyfit(log_freq, observation[:, j], poly_order)
        poly_fit = np.polyval(coeffs, log_freq)
        fg_estimate[:, j] = poly_fit
        residual[:, j] = observation[:, j] - poly_fit

    return residual, fg_estimate


def pca_foreground_removal(observation, n_components=N_PCA_COMPONENTS):
    """
    SVD-based foreground removal: leading singular modes capture the
    spectrally smooth foreground. Subtracting them reveals the 21cm signal.
    """
    U, S, Vt = np.linalg.svd(observation, full_matrices=False)
    fg_estimate = np.zeros_like(observation)
    for k in range(n_components):
        fg_estimate += S[k] * np.outer(U[:, k], Vt[k, :])
    residual = observation - fg_estimate
    return residual, fg_estimate


# ============================================================
# Metrics
# ============================================================

def compute_psnr(gt, recovered):
    data_range = np.max(gt) - np.min(gt)
    mse = np.mean((gt - recovered) ** 2)
    if mse == 0 or data_range == 0:
        return float('inf')
    return 10.0 * np.log10(data_range ** 2 / mse)


def compute_cc(gt, recovered):
    g = gt.ravel() - np.mean(gt)
    r = recovered.ravel() - np.mean(recovered)
    d = np.sqrt(np.sum(g**2) * np.sum(r**2))
    return float(np.sum(g * r) / d) if d > 0 else 0.0


def compute_rmse(gt, recovered):
    return float(np.sqrt(np.mean((gt - recovered) ** 2)))


# ============================================================
# Visualization
# ============================================================

def plot_results(frequencies, T21_gt, observation, T_fg_mK,
                 residual_poly, residual_pca, metrics):
    n_freq, n_angle = T21_gt.shape
    vmin_21 = np.percentile(T21_gt, 2)
    vmax_21 = np.percentile(T21_gt, 98)

    # ── Figure 1: Freq-Angle Maps ──
    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle('21cm Tomography: Foreground Removal Results', fontsize=16, y=0.98)

    kw = dict(aspect='auto', origin='lower',
              extent=[0, n_angle, FREQ_MIN, FREQ_MAX])

    im0 = axes[0, 0].imshow(T21_gt, cmap='RdBu_r', vmin=vmin_21, vmax=vmax_21, **kw)
    axes[0, 0].set_title('Ground Truth 21cm Signal')
    axes[0, 0].set_ylabel('Frequency (MHz)'); axes[0, 0].set_xlabel('Angular Pixel')
    plt.colorbar(im0, ax=axes[0, 0], label='T (mK)')

    im1 = axes[0, 1].imshow(observation, cmap='inferno', **kw)
    axes[0, 1].set_title('Observation (Signal+FG+Noise)')
    axes[0, 1].set_ylabel('Frequency (MHz)'); axes[0, 1].set_xlabel('Angular Pixel')
    plt.colorbar(im1, ax=axes[0, 1], label='T (mK)')

    im2 = axes[0, 2].imshow(T_fg_mK, cmap='inferno', **kw)
    axes[0, 2].set_title('Foreground (mK)')
    axes[0, 2].set_ylabel('Frequency (MHz)'); axes[0, 2].set_xlabel('Angular Pixel')
    plt.colorbar(im2, ax=axes[0, 2], label='T (mK)')

    im3 = axes[1, 0].imshow(residual_poly, cmap='RdBu_r', vmin=vmin_21, vmax=vmax_21, **kw)
    axes[1, 0].set_title(f'Poly Fit (order={POLY_ORDER})\n'
                         f'PSNR={metrics["poly_psnr"]:.1f} dB, CC={metrics["poly_cc"]:.4f}')
    axes[1, 0].set_ylabel('Frequency (MHz)'); axes[1, 0].set_xlabel('Angular Pixel')
    plt.colorbar(im3, ax=axes[1, 0], label='T (mK)')

    im4 = axes[1, 1].imshow(residual_pca, cmap='RdBu_r', vmin=vmin_21, vmax=vmax_21, **kw)
    axes[1, 1].set_title(f'PCA Recovery (n={N_PCA_COMPONENTS})\n'
                         f'PSNR={metrics["pca_psnr"]:.1f} dB, CC={metrics["pca_cc"]:.4f}')
    axes[1, 1].set_ylabel('Frequency (MHz)'); axes[1, 1].set_xlabel('Angular Pixel')
    plt.colorbar(im4, ax=axes[1, 1], label='T (mK)')

    error_pca = T21_gt - residual_pca
    im5 = axes[1, 2].imshow(error_pca, cmap='RdBu_r', **kw)
    axes[1, 2].set_title('Reconstruction Error (PCA)')
    axes[1, 2].set_ylabel('Frequency (MHz)'); axes[1, 2].set_xlabel('Angular Pixel')
    plt.colorbar(im5, ax=axes[1, 2], label='ΔT (mK)')

    plt.tight_layout()
    fig1.savefig(os.path.join(RESULTS_DIR, 'frequency_angle_maps.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # ── Figure 2: Spectral Profiles ──
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Spectral Profiles at Selected Angular Pixels', fontsize=14)
    pxs = [n_angle // 4, n_angle // 2, 3 * n_angle // 4, n_angle - 1]
    for ax, px in zip(axes2.ravel(), pxs):
        ax.plot(frequencies, T21_gt[:, px], 'k-', lw=2, label='GT 21cm', alpha=0.8)
        ax.plot(frequencies, residual_poly[:, px], 'b--', lw=1.5, label=f'Poly (ord={POLY_ORDER})', alpha=0.7)
        ax.plot(frequencies, residual_pca[:, px], 'r-.', lw=1.5, label=f'PCA (n={N_PCA_COMPONENTS})', alpha=0.7)
        ax.set_xlabel('Frequency (MHz)'); ax.set_ylabel('Temperature (mK)')
        ax.set_title(f'Pixel {px}'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(os.path.join(RESULTS_DIR, 'spectral_profiles.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # ── Figure 3: Power Spectra ──
    from numpy.fft import rfft, rfftfreq
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle('Power Spectrum Analysis', fontsize=14)

    kf = rfftfreq(n_freq, d=(frequencies[1] - frequencies[0]))
    for data, label, ls in [(T21_gt, 'GT', 'k-'), (residual_poly, 'Poly', 'b--'), (residual_pca, 'PCA', 'r-.')]:
        ax3a.semilogy(kf[1:], np.mean(np.abs(rfft(data, axis=0))**2, axis=1)[1:], ls, lw=1.5, label=label)
    ax3a.set_xlabel('Freq mode (1/MHz)'); ax3a.set_ylabel('Power')
    ax3a.set_title('Frequency Power Spectrum'); ax3a.legend(); ax3a.grid(True, alpha=0.3)

    ka = rfftfreq(n_angle)
    for data, label, ls in [(T21_gt, 'GT', 'k-'), (residual_poly, 'Poly', 'b--'), (residual_pca, 'PCA', 'r-.')]:
        ax3b.semilogy(ka[1:], np.mean(np.abs(rfft(data, axis=1))**2, axis=0)[1:], ls, lw=1.5, label=label)
    ax3b.set_xlabel('Angular mode'); ax3b.set_ylabel('Power')
    ax3b.set_title('Angular Power Spectrum'); ax3b.legend(); ax3b.grid(True, alpha=0.3)

    plt.tight_layout()
    fig3.savefig(os.path.join(RESULTS_DIR, 'power_spectra.png'), dpi=150, bbox_inches='tight')
    plt.close(fig3)

    print(f"  Figures saved to {RESULTS_DIR}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("21cm Tomography — Foreground Removal Inverse Problem")
    print("=" * 70)

    # 1. Ground Truth 21cm signal
    print("\n[1/6] Generating ground truth 21cm signal...")
    T21_gt = generate_21cm_signal(N_FREQ, N_ANGLE, rms=T21_RMS)
    print(f"  Shape: {T21_gt.shape}, Range: [{T21_gt.min():.2f}, {T21_gt.max():.2f}] mK, RMS: {np.std(T21_gt):.2f} mK")

    # 2. Foreground
    print("\n[2/6] Generating astrophysical foreground...")
    T_fg = generate_foreground(N_FREQ, N_ANGLE, FREQUENCIES)
    T_fg_mK = T_fg * 1000.0
    fg_ratio = np.std(T_fg_mK) / np.std(T21_gt)
    print(f"  Range: [{T_fg.min():.2f}, {T_fg.max():.2f}] K, FG/Signal ratio: {fg_ratio:.0f}x")

    # 3. Forward model
    print("\n[3/6] Forward model: observation = signal + foreground + noise...")
    observation, noise = forward_model(T21_gt, T_fg, noise_rms=NOISE_RMS)
    print(f"  Observation range: [{observation.min():.1f}, {observation.max():.1f}] mK")

    # 4. Polynomial fit
    print(f"\n[4/6] Polynomial foreground removal (order={POLY_ORDER}, log-freq)...")
    residual_poly, _ = polynomial_foreground_removal(observation, FREQUENCIES, POLY_ORDER)
    poly_psnr = compute_psnr(T21_gt, residual_poly)
    poly_cc   = compute_cc(T21_gt, residual_poly)
    poly_rmse = compute_rmse(T21_gt, residual_poly)
    print(f"  PSNR={poly_psnr:.2f} dB, CC={poly_cc:.4f}, RMSE={poly_rmse:.2f} mK")

    # 5. PCA
    print(f"\n[5/6] PCA foreground removal (n_components={N_PCA_COMPONENTS})...")
    residual_pca, _ = pca_foreground_removal(observation, N_PCA_COMPONENTS)
    pca_psnr = compute_psnr(T21_gt, residual_pca)
    pca_cc   = compute_cc(T21_gt, residual_pca)
    pca_rmse = compute_rmse(T21_gt, residual_pca)
    print(f"  PSNR={pca_psnr:.2f} dB, CC={pca_cc:.4f}, RMSE={pca_rmse:.2f} mK")

    metrics = {
        'poly_psnr': float(poly_psnr), 'poly_cc': float(poly_cc), 'poly_rmse': float(poly_rmse),
        'pca_psnr': float(pca_psnr),   'pca_cc': float(pca_cc),   'pca_rmse': float(pca_rmse),
        'signal_rms_mK': float(np.std(T21_gt)),
        'foreground_signal_ratio': float(fg_ratio),
        'noise_rms_mK': float(np.std(noise)),
        'n_freq': N_FREQ, 'n_angle': N_ANGLE,
        'freq_range_MHz': [float(FREQ_MIN), float(FREQ_MAX)],
    }

    # 6. Visualization
    print("\n[6/6] Generating visualizations...")
    plot_results(FREQUENCIES, T21_gt, observation, T_fg_mK,
                 residual_poly, residual_pca, metrics)

    metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Grid:              {N_FREQ} freq × {N_ANGLE} angle")
    print(f"  Freq range:        {FREQ_MIN}–{FREQ_MAX} MHz")
    print(f"  Signal RMS:        {np.std(T21_gt):.2f} mK")
    print(f"  FG/Signal ratio:   {fg_ratio:.0f}x")
    print(f"  Noise RMS:         {np.std(noise):.2f} mK")
    print(f"  ── Polynomial (order={POLY_ORDER}) ──")
    print(f"     PSNR = {poly_psnr:.2f} dB | CC = {poly_cc:.4f} | RMSE = {poly_rmse:.2f} mK")
    print(f"  ── PCA (n_comp={N_PCA_COMPONENTS}) ──")
    print(f"     PSNR = {pca_psnr:.2f} dB | CC = {pca_cc:.4f} | RMSE = {pca_rmse:.2f} mK")
    print("=" * 70)

    return metrics


if __name__ == '__main__':
    metrics = main()
