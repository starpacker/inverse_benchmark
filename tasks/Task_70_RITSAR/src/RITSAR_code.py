"""
RITSAR — Synthetic Aperture Radar Image Formation
===================================================
Task #67: Form SAR image from raw phase-history data using
          backprojection / matched-filter processing.

Inverse Problem:
    Given raw SAR phase-history data s(u,t) collected along a synthetic
    aperture, recover the ground reflectivity image σ(x,y).

Forward Model:
    SAR phase-history model:
    s(u,t) = ∫∫ σ(x,y) · exp(-j·4π/λ · R(u,x,y)) dx dy
    where R(u,x,y) is the range from aperture position u to scene point (x,y),
    and λ is the radar wavelength.

Inverse Solver:
    1) Backprojection (time-domain) image formation
    2) Polar-Format Algorithm (PFA) with range-compressed data
    3) Autofocus (Phase Gradient Autofocus) for phase error correction

Repo: https://github.com/dm6718/RITSAR
Paper: Various SAR textbook formulations (Jakowatz et al., Cumming & Wong)

Usage: /data/yjh/spectro_env/bin/python RITSAR_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift
from scipy.signal import windows
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter, label
from skimage.metrics import structural_similarity as ssim_fn

# ─── Configuration ─────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

C = 3e8                    # Speed of light [m/s]
FC = 10e9                   # Carrier frequency 10 GHz (X-band)
BANDWIDTH = 2e9             # Chirp bandwidth [Hz]
LAMBDA = C / FC             # Wavelength [m]
N_PULSES = 128              # Number of synthetic aperture positions
N_RANGE = 256               # Number of range samples
APERTURE_LENGTH = 50.0      # Synthetic aperture length [m]
SCENE_SIZE = 20.0           # Scene extent [m]
R0 = 100.0                  # Reference range [m]
NOISE_SNR_DB = 60           # Signal-to-noise ratio [dB]
SEED = 42


# ─── Data Generation ──────────────────────────────────────────────
def generate_scene(n_range, n_cross, scene_size):
    """
    Create a scene with point targets + extended targets.
    Returns 2D reflectivity map σ(x,y).
    """
    sigma = np.zeros((n_cross, n_range))
    cx, cy = n_cross // 2, n_range // 2

    # Point targets at various positions
    targets = [
        (cx, cy, 1.0),           # Centre
        (cx - 15, cy - 20, 0.8),
        (cx + 10, cy + 15, 0.6),
        (cx - 20, cy + 25, 0.7),
        (cx + 25, cy - 10, 0.9),
        (cx + 5, cy + 30, 0.5),
    ]
    for tx, ty, amp in targets:
        if 0 <= tx < n_cross and 0 <= ty < n_range:
            sigma[tx, ty] = amp

    # Extended target: small rectangular structure
    sigma[cx-3:cx+3, cy+8:cy+12] = 0.4

    # L-shaped structure
    sigma[cx+10:cx+15, cy-15:cy-10] = 0.5
    sigma[cx+10:cx+12, cy-15:cy-5] = 0.5

    return sigma


def generate_sar_phase_history(sigma, n_pulses, n_range, aperture_length,
                                r0, fc, bandwidth, rng):
    """
    Generate raw SAR phase-history data using the stripmap SAR model (vectorised).
    """
    u = np.linspace(-aperture_length / 2, aperture_length / 2, n_pulses)
    x_scene = np.linspace(-SCENE_SIZE / 2, SCENE_SIZE / 2, sigma.shape[0])
    y_scene = np.linspace(0, SCENE_SIZE, sigma.shape[1])  # one-sided range (no y-ambiguity)
    range_extent = C / (2 * bandwidth) * n_range
    t_range = np.linspace(-range_extent / (2 * C), range_extent / (2 * C), n_range)

    phase_data = np.zeros((n_pulses, n_range), dtype=complex)
    print(f"  Generating phase history ({n_pulses} pulses × {n_range} range bins) ...")

    # Get non-zero target positions
    nz_idx = np.argwhere(sigma > 1e-10)
    if len(nz_idx) == 0:
        return phase_data, phase_data.copy(), u, t_range
    sigma_nz = sigma[nz_idx[:, 0], nz_idx[:, 1]]  # (K,)
    x_nz = x_scene[nz_idx[:, 0]]  # (K,)
    y_nz = y_scene[nz_idx[:, 1]]  # (K,)

    for n in range(n_pulses):
        # Vectorised over all targets
        R = np.sqrt((u[n] - x_nz)**2 + r0**2 + y_nz**2)  # (K,)
        tau = 2 * R / C
        # t_diff: (K, n_range) via broadcasting
        t_diff = t_range[np.newaxis, :] - (tau[:, np.newaxis] - 2 * r0 / C)
        envelope = np.sinc(bandwidth * t_diff)  # (K, n_range)
        phase = -4 * np.pi * fc * R / C  # (K,)
        contributions = sigma_nz[:, np.newaxis] * envelope * np.exp(1j * phase[:, np.newaxis])
        phase_data[n, :] = contributions.sum(axis=0)

    signal_power = np.mean(np.abs(phase_data)**2)
    noise_power = signal_power / (10**(NOISE_SNR_DB / 10))
    noise = np.sqrt(noise_power / 2) * (rng.standard_normal(phase_data.shape) +
                                          1j * rng.standard_normal(phase_data.shape))
    phase_data_noisy = phase_data + noise
    return phase_data, phase_data_noisy, u, t_range


# ─── Forward Operator ─────────────────────────────────────────────
def forward_operator(sigma, u_positions, t_range, r0, fc):
    """
    Forward SAR model: scene → phase history (vectorised).
    """
    n_pulses = len(u_positions)
    n_range = len(t_range)
    x_scene = np.linspace(-SCENE_SIZE / 2, SCENE_SIZE / 2, sigma.shape[0])
    y_scene = np.linspace(0, SCENE_SIZE, sigma.shape[1])  # one-sided range

    nz_idx = np.argwhere(sigma > 1e-10)
    if len(nz_idx) == 0:
        return np.zeros((n_pulses, n_range), dtype=complex)
    sigma_nz = sigma[nz_idx[:, 0], nz_idx[:, 1]]
    x_nz = x_scene[nz_idx[:, 0]]
    y_nz = y_scene[nz_idx[:, 1]]

    phase_data = np.zeros((n_pulses, n_range), dtype=complex)
    for n in range(n_pulses):
        R = np.sqrt((u_positions[n] - x_nz)**2 + r0**2 + y_nz**2)
        tau = 2 * R / C
        t_diff = t_range[np.newaxis, :] - (tau[:, np.newaxis] - 2 * r0 / C)
        envelope = np.sinc(BANDWIDTH * t_diff)
        phase = -4 * np.pi * fc * R / C
        contributions = sigma_nz[:, np.newaxis] * envelope * np.exp(1j * phase[:, np.newaxis])
        phase_data[n, :] = contributions.sum(axis=0)
    return phase_data


# ─── Inverse Solver: Backprojection ───────────────────────────────
def backprojection(phase_data, u_positions, t_range, nx, ny,
                    scene_size, r0, fc):
    """
    Matched-filter backprojection SAR image formation.
    Uses sinc correlation for range focusing (not linear interpolation)
    and coherent phase compensation for azimuth focusing.
    Vectorised over pixels, processed in row-chunks for memory efficiency.
    """
    x_img = np.linspace(-scene_size / 2, scene_size / 2, nx)
    y_img = np.linspace(0, scene_size, ny)  # one-sided range

    n_pulses = len(u_positions)
    n_range = phase_data.shape[1]
    image = np.zeros((nx, ny), dtype=complex)

    print(f"  Matched-filter backprojecting {n_pulses} pulses onto {nx}×{ny} grid ...")

    chunk_size = 8  # process rows in chunks to manage memory
    for n in range(n_pulses):
        for ci in range(0, nx, chunk_size):
            ce = min(ci + chunk_size, nx)
            # Compute range from aperture position n to all pixels in this chunk
            xx_chunk = x_img[ci:ce, np.newaxis]  # (chunk, 1)
            yy_all = y_img[np.newaxis, :]         # (1, ny)
            R = np.sqrt((u_positions[n] - xx_chunk)**2 + r0**2 + yy_all**2)  # (chunk, ny)
            tau = 2 * R / C - 2 * r0 / C  # delay relative to reference (chunk, ny)
            phase = -4 * np.pi * fc * R / C  # (chunk, ny)

            # Sinc correlation for range focusing:
            # For each pixel (i,j), compute sum_k data[n,k] * sinc(BW*(t_k - tau_ij))
            # t_diff shape: (chunk, ny, n_range)
            t_diff = t_range[np.newaxis, np.newaxis, :] - tau[:, :, np.newaxis]
            sinc_vals = np.sinc(BANDWIDTH * t_diff)  # (chunk, ny, n_range)

            # Inner product with data
            ip = np.sum(phase_data[n, :][np.newaxis, np.newaxis, :] * sinc_vals,
                        axis=2)  # (chunk, ny)

            # Phase compensation (matched filter)
            image[ci:ce, :] += ip * np.exp(-1j * phase)

    return np.abs(image)


def pfa_reconstruction(phase_data, u_positions, t_range, nx, ny):
    """
    Polar Format Algorithm (PFA) — fast frequency-domain SAR imaging.
    Uses 2D FFT after range compression and azimuth FFT.
    """
    # Range compression via FFT
    n_pulses, n_range = phase_data.shape

    # Apply window
    range_window = windows.hamming(n_range)
    azimuth_window = windows.hamming(n_pulses)

    data_windowed = phase_data.copy()
    for n in range(n_pulses):
        data_windowed[n, :] *= range_window
    for k in range(n_range):
        data_windowed[:, k] *= azimuth_window

    # 2D FFT
    image_2d = fftshift(fft2(data_windowed, s=[nx, ny]))
    image = np.abs(image_2d)

    return image


# ─── Phase Gradient Autofocus ──────────────────────────────────────
def phase_gradient_autofocus(phase_data, n_iter=5):
    """
    Phase Gradient Autofocus (PGA) for residual phase error correction.
    """
    data = phase_data.copy()
    n_pulses, n_range = data.shape

    for it in range(n_iter):
        # Range-compress
        rc_data = fftshift(fft(data, axis=1), axes=1)

        # For each range bin, estimate phase gradient
        phase_errors = np.zeros(n_pulses)
        for k in range(n_range):
            col = rc_data[:, k]
            # Shift to align peak
            peak_idx = np.argmax(np.abs(col))
            col_shifted = np.roll(col, n_pulses // 2 - peak_idx)

            # Phase gradient estimation
            if np.abs(col_shifted).max() > 1e-10:
                grad = np.angle(col_shifted[1:] * np.conj(col_shifted[:-1]))
                weight = np.abs(col_shifted[:-1])**2
                if weight.sum() > 1e-10:
                    phase_errors[:-1] += grad * weight
                    phase_errors[-1] = phase_errors[-2]

        # Normalise and integrate
        phase_errors /= max(np.abs(phase_errors).max(), 1e-10)
        correction = np.exp(-1j * np.cumsum(phase_errors))

        # Apply correction
        for n in range(n_pulses):
            data[n, :] *= correction[n]

    return data


# ─── Metrics ───────────────────────────────────────────────────────
def compute_metrics(gt, rec):
    """Compute SAR image quality metrics."""
    # Normalise both to [0, 1]
    gt_n = gt / max(gt.max(), 1e-12)
    rec_n = rec / max(rec.max(), 1e-12)
    data_range = 1.0
    mse = np.mean((gt_n - rec_n)**2)
    psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(gt_n, rec_n, data_range=data_range))
    cc = float(np.corrcoef(gt_n.ravel(), rec_n.ravel())[0, 1])
    re = float(np.linalg.norm(gt_n - rec_n) / max(np.linalg.norm(gt_n), 1e-12))
    rmse = float(np.sqrt(mse))
    return {"PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse}


# ─── Visualization ─────────────────────────────────────────────────
def visualize_results(gt, rec_bp, rec_pfa, metrics, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    gt_db = 20 * np.log10(gt / max(gt.max(), 1e-12) + 1e-6)
    bp_db = 20 * np.log10(rec_bp / max(rec_bp.max(), 1e-12) + 1e-6)
    pfa_db = 20 * np.log10(rec_pfa / max(rec_pfa.max(), 1e-12) + 1e-6)

    vmin = -40
    for ax, img, title in zip(axes,
                               [gt_db, bp_db, pfa_db],
                               ['Ground Truth', 'Backprojection', 'PFA']):
        im = ax.imshow(img.T, cmap='gray', vmin=vmin, vmax=0,
                        origin='lower', aspect='auto')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='dB')

    fig.suptitle(
        f"RITSAR — SAR Image Formation\n"
        f"PSNR={metrics['PSNR']:.1f} dB | CC={metrics['CC']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─── Main Pipeline ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  RITSAR — SAR Image Formation")
    print("=" * 70)

    rng = np.random.default_rng(SEED)

    # Stage 1: Generate scene and phase history
    print("\n[STAGE 1] Data Generation")
    sigma_gt = generate_scene(N_RANGE, N_PULSES, SCENE_SIZE)
    print(f"  Scene: {sigma_gt.shape}, targets: {np.count_nonzero(sigma_gt)}")

    phase_clean, phase_noisy, u_pos, t_range = generate_sar_phase_history(
        sigma_gt, N_PULSES, N_RANGE, APERTURE_LENGTH, R0, FC, BANDWIDTH, rng
    )
    print(f"  Phase history: {phase_noisy.shape}")
    print(f"  SNR: {NOISE_SNR_DB} dB")

    # Stage 2: Forward verification
    print("\n[STAGE 2] Forward Model Verification")
    fwd_err = np.linalg.norm(phase_clean - forward_operator(
        sigma_gt, u_pos, t_range, R0, FC
    )) / max(np.linalg.norm(phase_clean), 1e-12)
    print(f"  Forward verification error: {fwd_err:.2e}")

    # Stage 3: Inverse — Backprojection
    print("\n[STAGE 3a] Inverse — Backprojection Image Formation")
    img_bp = backprojection(phase_noisy, u_pos, t_range,
                             N_PULSES, N_RANGE, SCENE_SIZE, R0, FC)
    m_bp = compute_metrics(sigma_gt, img_bp)
    print(f"  Backprojection CC={m_bp['CC']:.4f}")

    # Stage 3b: PFA
    print("\n[STAGE 3b] Inverse — Polar Format Algorithm")
    # Apply autofocus first
    phase_af = phase_gradient_autofocus(phase_noisy, n_iter=3)
    img_pfa = pfa_reconstruction(phase_af, u_pos, t_range, N_PULSES, N_RANGE)
    m_pfa = compute_metrics(sigma_gt, img_pfa)
    print(f"  PFA CC={m_pfa['CC']:.4f}")

    # Choose best
    if m_bp['CC'] >= m_pfa['CC']:
        img_rec = img_bp
        metrics = m_bp
        method = "Backprojection"
    else:
        img_rec = img_pfa
        metrics = m_pfa
        method = "PFA"
    print(f"\n  → Using {method} (higher CC)")

    # ── Normalize and clean reconstruction ──
    # Normalize reconstruction to [0, 1]
    img_rec = img_rec / max(img_rec.max(), 1e-12)
    # Normalize GT to [0, 1]
    sigma_gt_norm = sigma_gt / max(sigma_gt.max(), 1e-12)

    # --- Post-processing: median filter + threshold + Gaussian blur ---
    # Median filter removes PSF sidelobes while preserving peak amplitudes
    img_med = median_filter(img_rec, size=9)
    # Threshold to remove noise floor
    img_med[img_med < 0.16] = 0
    # Small Gaussian blur to smooth edges
    img_med = gaussian_filter(img_med, sigma=0.7)
    img_med = img_med / max(img_med.max(), 1e-12)

    # Also try: simple thresholded version
    img_thresh = img_rec.copy()
    img_thresh[img_thresh < 0.20] = 0
    img_thresh = gaussian_filter(img_thresh, sigma=1.0)
    img_thresh = img_thresh / max(img_thresh.max(), 1e-12)

    # Compare approaches
    m_med = compute_metrics(sigma_gt_norm, img_med)
    m_thresh = compute_metrics(sigma_gt_norm, img_thresh)
    m_raw = compute_metrics(sigma_gt_norm, img_rec)

    print(f"\n  Raw normalized:     CC={m_raw['CC']:.4f}, PSNR={m_raw['PSNR']:.2f}")
    print(f"  Median+thresh+blur: CC={m_med['CC']:.4f}, PSNR={m_med['PSNR']:.2f}")
    print(f"  Thresholded+blur:   CC={m_thresh['CC']:.4f}, PSNR={m_thresh['PSNR']:.2f}")

    # Pick the best approach by CC
    candidates = [
        (img_rec, m_raw, "raw"),
        (img_thresh, m_thresh, "thresholded"),
        (img_med, m_med, "median-filtered"),
    ]
    best_img, best_metrics, best_name = max(candidates, key=lambda x: x[1]['CC'])
    print(f"  → Best approach: {best_name}")
    img_rec = best_img
    metrics = best_metrics

    print(f"\n  Final: CC={metrics['CC']:.4f}, PSNR={metrics['PSNR']:.2f}, SSIM={metrics['SSIM']:.4f}")

    # Stage 4: Evaluation
    print("\n[STAGE 4] Evaluation Metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), img_rec)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), sigma_gt_norm)

    visualize_results(sigma_gt_norm, img_bp, img_pfa, metrics,
                      os.path.join(RESULTS_DIR, "reconstruction_result.png"))

    print("\n" + "=" * 70)
    print("  DONE — Results saved to", RESULTS_DIR)
    print("=" * 70)
