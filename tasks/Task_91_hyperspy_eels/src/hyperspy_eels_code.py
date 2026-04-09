"""
EELS Deconvolution via Fourier-Log Method

Inverse Problem:
    Recover the single scattering distribution (SSD) from a measured electron
    energy loss spectrum (EELS) that contains multiple scattering contributions.

Forward Model (Poisson multiple scattering):
    In Fourier domain:
        F[J](q) = F[Z](q) * exp( (t/lambda) * F[S_norm](q) )

    where Z is the zero-loss peak, S_norm is the normalized single scattering
    distribution (integral = 1), and t/lambda is the relative specimen thickness.

    In energy domain this corresponds to the Poisson series:
        J(E) = Z(E) * [ delta(E) + (t/l)*S(E) + (t/l)^2/(2!)*S*S(E) + ... ]

Inverse Solver (Fourier-Log Deconvolution):
    (t/lambda) * S_norm(E) = F^{-1}[ ln( F[J] / F[Z] ) ]

    We apply a frequency-domain taper that smoothly rolls off the contribution
    from frequencies where |F[Z]| is small relative to its peak, to prevent
    noise amplification in the deconvolution.

Reference:
    R.F. Egerton, "Electron Energy-Loss Spectroscopy in the Electron Microscope",
    3rd Edition, Springer, 2011. Chapter 4.
"""

import matplotlib
matplotlib.use('Agg')

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft


# =============================================================================
# Signal Generation
# =============================================================================

def generate_zlp(n_channels, de, fwhm):
    """
    Generate a zero-loss peak (ZLP) as a Gaussian centered at channel 0.

    The ZLP wraps around: negative-energy channels are at the end of the array.
    Normalized to sum = 1.

    Parameters
    ----------
    n_channels : int
        Number of spectral channels.
    de : float
        Energy per channel (eV).
    fwhm : float
        Full width at half maximum in eV.

    Returns
    -------
    zlp : ndarray of shape (n_channels,)
    """
    sigma_ch = (fwhm / de) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    idx = np.arange(n_channels)
    dist = np.minimum(idx, n_channels - idx).astype(float)
    zlp = np.exp(-dist**2 / (2.0 * sigma_ch**2))
    zlp /= np.sum(zlp)
    return zlp


def generate_ground_truth_ssd(n_channels, de):
    """
    Generate a ground truth single scattering distribution (SSD).

    Simulates typical EELS plasmon features for aluminum-like material:
    - Surface plasmon peak at ~7 eV
    - Bulk plasmon peak at ~15 eV (dominant)
    - Interband transitions at ~30 eV
    - Weak high-energy feature at ~50 eV

    Returns normalized SSD (sum = 1) and the raw SSD.

    Parameters
    ----------
    n_channels : int
    de : float
        Energy per channel (eV).

    Returns
    -------
    ssd_norm : ndarray
        Normalized SSD (sum = 1).
    ssd_raw : ndarray
        Unnormalized SSD.
    """
    idx = np.arange(n_channels)

    peaks = [
        (7.0, 2.0, 0.30),   # surface plasmon: (center_eV, sigma_eV, amplitude)
        (15.0, 3.0, 0.80),  # bulk plasmon (dominant)
        (30.0, 5.0, 0.20),  # interband transitions
        (50.0, 4.0, 0.10),  # weak high-energy feature
    ]

    ssd = np.zeros(n_channels)
    for center_eV, sigma_eV, amp in peaks:
        c = center_eV / de
        s = sigma_eV / de
        ssd += amp * np.exp(-(idx - c)**2 / (2.0 * s**2))

    # Enforce causality: zero for E < 2 eV and E > 100 eV
    ssd[idx < int(2.0 / de)] = 0.0
    ssd[idx > int(100.0 / de)] = 0.0

    ssd_norm = ssd / (np.sum(ssd) + 1e-30)
    return ssd_norm, ssd


# =============================================================================
# Forward Model
# =============================================================================

def forward_multiple_scattering(zlp, ssd_normalized, t_over_lambda):
    """
    Forward model: simulate measured EELS with Poisson multiple scattering.

        F[J] = F[Z] * exp( t/lambda * F[S_norm] )

    Parameters
    ----------
    zlp : ndarray
        Zero-loss peak (sum = 1, centered at channel 0).
    ssd_normalized : ndarray
        Normalized SSD (sum = 1).
    t_over_lambda : float
        Relative thickness.

    Returns
    -------
    measured : ndarray
        Simulated measured EELS spectrum (non-negative).
    """
    Z = fft(zlp)
    S = fft(ssd_normalized)
    J_ft = Z * np.exp(t_over_lambda * S)
    measured = np.real(ifft(J_ft))
    return np.maximum(measured, 0.0)


# =============================================================================
# Inverse Solver
# =============================================================================

def fourier_log_deconvolution(measured_spectrum, zlp, taper_threshold=0.02):
    """
    Fourier-log deconvolution with frequency-domain tapering.

    Algorithm:
        1. Compute ratio R(q) = F[J](q) / F[Z](q) with regularization.
        2. Apply smooth taper T(q) that rolls off for |F[Z](q)| < threshold.
        3. Recovered SSD = F^{-1}[ T(q) * ln(R(q)) ]

    The taper prevents noise amplification at frequencies where the ZLP's
    Fourier transform is vanishingly small.

    Parameters
    ----------
    measured_spectrum : ndarray
        Measured EELS spectrum.
    zlp : ndarray
        Zero-loss peak.
    taper_threshold : float
        Frequencies with |F[Z]| / max(|F[Z]|) below this are tapered to zero.

    Returns
    -------
    ssd_recovered : ndarray
        Recovered SSD = (t/lambda) * S_norm.
    """
    J = fft(measured_spectrum)
    Z = fft(zlp)

    # Frequency-domain taper: smooth rolloff where |Z| is small
    Z_abs_norm = np.abs(Z) / np.max(np.abs(Z))
    taper = np.clip(Z_abs_norm / taper_threshold, 0.0, 1.0)

    # Regularized division
    Z_reg = Z + 1e-10 * np.max(np.abs(Z))
    ratio = J / Z_reg

    # Complex logarithm with taper
    log_ratio = (np.log(np.abs(ratio) + 1e-30) + 1j * np.angle(ratio)) * taper

    ssd_recovered = np.real(ifft(log_ratio))
    return ssd_recovered


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(ground_truth, reconstruction):
    """
    Compute quality metrics for 1D spectral data.

    Parameters
    ----------
    ground_truth, reconstruction : ndarray
        Arrays to compare (same length, region of interest).

    Returns
    -------
    dict with keys: PSNR, RMSE, CC, relative_error
    """
    gt = ground_truth.astype(np.float64)
    rec = reconstruction.astype(np.float64)

    mse = np.mean((gt - rec)**2)
    rmse = np.sqrt(mse)

    data_range = np.max(gt) - np.min(gt)
    if data_range > 0 and rmse > 0:
        psnr = 20.0 * np.log10(data_range / rmse)
    else:
        psnr = float('inf')

    gt_c = gt - np.mean(gt)
    rec_c = rec - np.mean(rec)
    denom = np.sqrt(np.sum(gt_c**2) * np.sum(rec_c**2))
    cc = float(np.sum(gt_c * rec_c) / denom) if denom > 0 else 0.0

    gt_norm = np.linalg.norm(gt)
    rel_err = float(np.linalg.norm(gt - rec) / gt_norm) if gt_norm > 0 else float('inf')

    return {
        "PSNR": float(np.round(psnr, 4)),
        "RMSE": float(np.round(rmse, 6)),
        "CC": float(np.round(cc, 6)),
        "relative_error": float(np.round(rel_err, 6))
    }


# =============================================================================
# Visualization
# =============================================================================

def visualize_results(energy_eV, gt_ssd, rec_ssd, measured, zlp, metrics, path):
    """
    Create 3-panel figure:
      1. Measured EELS spectrum with ZLP overlay
      2. Ground truth vs reconstructed SSD
      3. Residual plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10),
                             gridspec_kw={'height_ratios': [1.2, 1.2, 0.8]})
    xl = [0, 80]

    # --- Panel 1: Measured EELS ---
    ax = axes[0]
    ax.plot(energy_eV, measured, 'b-', lw=1.0, alpha=0.8,
            label='Measured EELS (multiple scattering)')
    scale = 0.3 * np.max(measured) / (np.max(zlp) + 1e-30)
    ax.plot(energy_eV, zlp * scale, 'g--', lw=1.0, alpha=0.7,
            label='Zero-Loss Peak (scaled)')
    ax.set_xlabel('Energy Loss (eV)', fontsize=11)
    ax.set_ylabel('Intensity (a.u.)', fontsize=11)
    ax.set_title('EELS Measurement with Multiple Scattering',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(xl)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: GT vs Reconstructed SSD ---
    ax = axes[1]
    ax.plot(energy_eV, gt_ssd, 'k-', lw=2.0, label='Ground Truth SSD')
    ax.plot(energy_eV, rec_ssd, 'r--', lw=1.5, alpha=0.9,
            label='Reconstructed SSD (Fourier-Log)')
    ax.set_xlabel('Energy Loss (eV)', fontsize=11)
    ax.set_ylabel('Intensity (a.u.)', fontsize=11)
    ax.set_title(
        f'Single Scattering Distribution Recovery\n'
        f'PSNR = {metrics["PSNR"]:.2f} dB | CC = {metrics["CC"]:.4f} | '
        f'RMSE = {metrics["RMSE"]:.4e}',
        fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(xl)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Residual ---
    ax = axes[2]
    residual = gt_ssd - rec_ssd
    ax.plot(energy_eV, residual, 'purple', lw=1.0, alpha=0.8)
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.fill_between(energy_eV, residual, alpha=0.2, color='purple')
    ax.set_xlabel('Energy Loss (eV)', fontsize=11)
    ax.set_ylabel('Residual', fontsize=11)
    ax.set_title('Residual (GT − Reconstruction)', fontsize=13, fontweight='bold')
    ax.set_xlim(xl)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {path}")


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    """
    Complete pipeline:
      1. Generate synthetic EELS data (ZLP + SSD)
      2. Forward model: Poisson multiple scattering
      3. Inverse: Fourier-log deconvolution
      4. Compute metrics
      5. Save results and visualization
    """
    # ===== Configuration =====
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Spectrometer parameters
    n_channels = 512         # Number of spectral channels
    de = 0.5                 # Energy dispersion: eV per channel
    zlp_fwhm = 2.0           # ZLP FWHM in eV (typical for Schottky emitter)

    # Physical parameters
    t_over_lambda = 0.8      # Relative thickness (moderate multiple scattering)
    noise_level = 0.0005     # Relative noise amplitude (good counting statistics)

    energy_axis = np.arange(n_channels) * de  # eV

    print("=" * 60)
    print("EELS Deconvolution via Fourier-Log Method")
    print("=" * 60)
    print(f"Channels: {n_channels}, dE = {de} eV/ch")
    print(f"Energy range: 0 – {n_channels * de:.1f} eV")
    print(f"Relative thickness t/λ = {t_over_lambda}")
    print(f"ZLP FWHM = {zlp_fwhm} eV")
    print(f"Noise level = {noise_level}")

    # ===== Step 1: Generate signals =====
    zlp = generate_zlp(n_channels, de, zlp_fwhm)
    ssd_norm, _ = generate_ground_truth_ssd(n_channels, de)

    # Ground truth: what the deconvolution should recover
    gt_ssd = t_over_lambda * ssd_norm

    print(f"\nZLP sum = {np.sum(zlp):.6f}")
    print(f"SSD_norm sum = {np.sum(ssd_norm):.6f}")
    print(f"GT SSD (t/λ · S_norm) max = {np.max(gt_ssd):.6f}")

    # ===== Step 2: Forward model =====
    print("\n--- Forward Model: Multiple Scattering ---")
    measured_clean = forward_multiple_scattering(zlp, ssd_norm, t_over_lambda)

    # Add Gaussian noise
    rng = np.random.default_rng(42)
    noise = noise_level * np.max(measured_clean) * rng.standard_normal(n_channels)
    measured = np.maximum(measured_clean + noise, 0.0)

    print(f"Measured max = {np.max(measured):.6f}")
    print(f"Measured sum = {np.sum(measured):.6f}")

    # ===== Step 3: Fourier-log deconvolution =====
    print("\n--- Inverse: Fourier-Log Deconvolution ---")
    rec_ssd = fourier_log_deconvolution(measured, zlp, taper_threshold=0.02)

    # Enforce causality: zero outside physical region
    rec_ssd[energy_axis < 2.0] = 0.0
    rec_ssd[energy_axis > 100.0] = 0.0

    print(f"Reconstructed SSD max = {np.max(rec_ssd):.6f}")
    print(f"Reconstructed SSD sum = {np.sum(rec_ssd):.6f}")

    # ===== Step 4: Compute metrics (ROI: 2–80 eV) =====
    roi = (energy_axis >= 2.0) & (energy_axis <= 80.0)
    metrics = compute_metrics(gt_ssd[roi], rec_ssd[roi])

    print("\n--- Metrics (ROI: 2–80 eV) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # ===== Step 5: Save results =====
    print("\n--- Saving Results ---")

    with open(os.path.join(results_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("  metrics.json saved")

    np.save(os.path.join(results_dir, "ground_truth.npy"), gt_ssd)
    np.save(os.path.join(results_dir, "reconstruction.npy"), rec_ssd)
    np.save(os.path.join(results_dir, "input_measurement.npy"), measured)
    np.save(os.path.join(results_dir, "energy_axis.npy"), energy_axis)
    print("  .npy files saved")

    # ===== Step 6: Visualization =====
    vis_path = os.path.join(results_dir, "reconstruction_result.png")
    visualize_results(energy_axis, gt_ssd, rec_ssd, measured, zlp, metrics, vis_path)

    print("\n" + "=" * 60)
    print("EELS Fourier-Log Deconvolution — COMPLETE")
    print(f"Results: {results_dir}")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    main()
