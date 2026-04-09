"""
nmrglue — NMR Spectrum Reconstruction Inverse Problem
======================================================
Task: Reconstruct a 2D NMR spectrum from non-uniformly sampled (NUS)
      free induction decay (FID) data.

Inverse Problem:
    Given sparsely sampled time-domain FID data S(t₁, t₂) at a subset
    of t₁ points, reconstruct the full 2D frequency-domain spectrum
    I(ω₁, ω₂).

Forward Model (nmrglue):
    I(ω₁, ω₂) = FFT₂D{ S(t₁, t₂) · W(t₁, t₂) }
    where S is the complete FID and W is the NUS sampling mask.
    nmrglue handles Bruker/Varian/Agilent data formats, apodization,
    zero-filling, phasing, and spectral processing.

Inverse Solver:
    Iterative Soft Thresholding (IST) / Compressed Sensing
    reconstruction from NUS data using L1 minimisation in
    the frequency domain.

Repo: https://github.com/jjhelmus/nmrglue
Paper: Helmus & Jaroniec (2013), J. Biomol. NMR, 55, 355–367.
       doi:10.1007/s10858-013-9718-x

Usage:
    /data/yjh/spectro_env/bin/python nmrglue_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.fft import fft, ifft, fft2, ifft2, fftshift
from skimage.metrics import structural_similarity as ssim_fn

# ── nmrglue library import ──────────────────────────────────
import nmrglue as ng

# ═══════════════════════════════════════════════════════════
# 1. Configuration
# ═══════════════════════════════════════════════════════════
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Synthetic 2D NMR parameters
N_F2 = 512       # direct dimension points (fully sampled)
N_F1 = 256       # indirect dimension points (ground truth)
NUS_FRAC = 0.25  # fraction of indirect-dimension points sampled (25% NUS)
N_PEAKS = 12     # number of resonance peaks
SW_F2 = 12000.0  # spectral width F2 [Hz]
SW_F1 = 8000.0   # spectral width F1 [Hz]
OBS_F2 = 600.0   # observe frequency F2 [MHz]
OBS_F1 = 150.0   # observe frequency F1 [MHz]
NOISE_STD = 0.02 # noise level (relative to max FID amplitude)
SEED = 42

# IST reconstruction parameters
IST_ITERATIONS = 200
IST_THRESHOLD_DECAY = 0.98  # threshold decrease per iteration


# ═══════════════════════════════════════════════════════════
# 2. Synthetic NMR Signal Generation (using nmrglue processing)
# ═══════════════════════════════════════════════════════════
def generate_synthetic_peaks(n_peaks, seed=42):
    """
    Generate random peak parameters for a 2D NMR spectrum.

    Returns list of dicts with: freq_f1, freq_f2, lw_f1, lw_f2, amplitude, phase
    """
    rng = np.random.default_rng(seed)

    peaks = []
    for i in range(n_peaks):
        peaks.append({
            "freq_f1": rng.uniform(0.15, 0.85) * SW_F1,   # Hz
            "freq_f2": rng.uniform(0.15, 0.85) * SW_F2,   # Hz
            "lw_f1": rng.uniform(10, 50),                  # Hz linewidth
            "lw_f2": rng.uniform(15, 80),                  # Hz linewidth
            "amplitude": rng.uniform(0.5, 2.0),
            "phase": rng.uniform(-0.1, 0.1),               # small phase error
        })
    return peaks


def synthesize_fid(peaks, n_f1, n_f2, sw_f1, sw_f2):
    """
    Synthesize a 2D FID from Lorentzian peaks.

    S(t1, t2) = Σ_k A_k · exp(i·2π·ν₁_k·t1 - π·Δν₁_k·t1)
                         · exp(i·2π·ν₂_k·t2 - π·Δν₂_k·t2)
    """
    dt1 = 1.0 / sw_f1
    dt2 = 1.0 / sw_f2
    t1 = np.arange(n_f1) * dt1
    t2 = np.arange(n_f2) * dt2

    fid = np.zeros((n_f1, n_f2), dtype=complex)
    for p in peaks:
        decay_f1 = np.exp(-np.pi * p["lw_f1"] * t1)
        osc_f1 = np.exp(1j * 2 * np.pi * p["freq_f1"] * t1)
        decay_f2 = np.exp(-np.pi * p["lw_f2"] * t2)
        osc_f2 = np.exp(1j * 2 * np.pi * p["freq_f2"] * t2)
        sig_f1 = p["amplitude"] * np.exp(1j * p["phase"]) * decay_f1 * osc_f1
        sig_f2 = decay_f2 * osc_f2
        fid += np.outer(sig_f1, sig_f2)

    return fid


def forward_operator(fid_full, nus_schedule):
    """
    NUS forward operator: apply sampling mask to FID.

    Uses nmrglue's processing pipeline for apodization and
    zero-filling of the direct dimension, then masks the
    indirect dimension according to the NUS schedule.

    Parameters
    ----------
    fid_full : np.ndarray  Complete 2D FID (n_f1 × n_f2).
    nus_schedule : np.ndarray  Boolean mask of sampled t1 points.

    Returns
    -------
    fid_nus : np.ndarray  NUS-sampled FID (same shape, zeros at
                          unsampled t1 rows).
    """
    # Process direct dimension (F2) with nmrglue
    dic = ng.bruker.guess_udic({"acqus": {"SW_h": SW_F2, "SFO1": OBS_F2}}, fid_full)

    # Apply nmrglue apodization to F2 (direct dimension)
    fid_proc = ng.proc_base.em(fid_full, lb=5.0)  # 5 Hz exponential line broadening

    # Zero-fill F2 if needed
    fid_proc = ng.proc_base.zf_size(fid_proc, fid_proc.shape[0], N_F2)

    # Apply NUS mask to indirect dimension (F1)
    fid_nus = np.zeros_like(fid_proc)
    fid_nus[nus_schedule, :] = fid_proc[nus_schedule, :]

    return fid_nus


# ═══════════════════════════════════════════════════════════
# 3. Data Generation
# ═══════════════════════════════════════════════════════════
def load_or_generate_data():
    """Generate synthetic NUS NMR data."""
    print("[DATA] Generating synthetic 2D NMR peaks ...")
    peaks = generate_synthetic_peaks(N_PEAKS, SEED)

    print(f"[DATA] Synthesising complete FID ({N_F1}×{N_F2}) ...")
    fid_full = synthesize_fid(peaks, N_F1, N_F2, SW_F1, SW_F2)

    # Ground truth spectrum (full FFT with nmrglue processing)
    fid_proc = ng.proc_base.em(fid_full, lb=5.0)
    spec_gt = fftshift(fft2(fid_proc)).real
    # Normalise
    spec_gt = spec_gt / np.abs(spec_gt).max()

    # NUS schedule (random sampling of indirect dimension)
    rng = np.random.default_rng(SEED + 1)
    n_sampled = max(int(N_F1 * NUS_FRAC), 2)
    # Always include first point
    schedule = np.zeros(N_F1, dtype=bool)
    schedule[0] = True
    chosen = rng.choice(np.arange(1, N_F1), size=n_sampled - 1, replace=False)
    schedule[chosen] = True

    print(f"[DATA] NUS schedule: {schedule.sum()}/{N_F1} points "
          f"({schedule.sum()/N_F1*100:.0f}%)")

    # Apply NUS forward operator (with noise)
    rng2 = np.random.default_rng(SEED + 2)
    noise = NOISE_STD * np.abs(fid_full).max() * (
        rng2.standard_normal(fid_full.shape) +
        1j * rng2.standard_normal(fid_full.shape)
    )
    fid_noisy = fid_full + noise
    fid_nus = forward_operator(fid_noisy, schedule)

    return fid_nus, spec_gt, fid_full, schedule


# ═══════════════════════════════════════════════════════════
# 4. Inverse Solver — Iterative Soft Thresholding (IST)
# ═══════════════════════════════════════════════════════════
def soft_threshold(x, thresh):
    """Complex soft thresholding."""
    mag = np.abs(x)
    return np.where(mag > thresh, x * (1 - thresh / np.maximum(mag, 1e-30)), 0)


def reconstruct(fid_nus, schedule):
    """
    Reconstruct 2D NMR spectrum from NUS data using IST
    (Iterative Soft Thresholding).

    Uses nmrglue for processing the direct dimension (apodization,
    FFT) and IST for the indirect dimension.

    Parameters
    ----------
    fid_nus : np.ndarray   NUS-sampled FID.
    schedule : np.ndarray  Boolean NUS sampling mask.

    Returns
    -------
    spec_recon : np.ndarray  Reconstructed 2D spectrum.
    """
    print(f"[RECON] IST reconstruction ({IST_ITERATIONS} iterations) ...")

    # Process F2 (direct) dimension: apodization + FFT via nmrglue
    fid_f2 = ng.proc_base.em(fid_nus, lb=5.0)
    data_f2 = np.zeros_like(fid_f2, dtype=complex)
    for i in range(fid_f2.shape[0]):
        data_f2[i, :] = fft(fid_f2[i, :])

    # IST on F1 (indirect) dimension
    # Initial estimate: zero-filled FT
    current = data_f2.copy()

    # Initial threshold from max of zero-filled spectrum
    zf_spec = fftshift(fft2(fid_f2)).real
    thresh = 0.99 * np.abs(zf_spec).max()

    for it in range(IST_ITERATIONS):
        # Step 1: FFT along F1
        spec = np.zeros_like(current, dtype=complex)
        for j in range(current.shape[1]):
            spec[:, j] = fft(current[:, j])

        # Step 2: Soft threshold in frequency domain
        spec_thresh = soft_threshold(spec, thresh)

        # Step 3: iFFT back to time domain
        for j in range(current.shape[1]):
            current[:, j] = ifft(spec_thresh[:, j])

        # Step 4: Enforce data consistency at sampled points
        current[schedule, :] = data_f2[schedule, :]

        # Decay threshold
        thresh *= IST_THRESHOLD_DECAY

        if (it + 1) % 50 == 0:
            residual = np.linalg.norm(current[schedule, :] - data_f2[schedule, :])
            print(f"[RECON]   iter {it+1:4d}  thresh={thresh:.4e}  "
                  f"residual={residual:.4e}")

    # Final spectrum
    spec_recon = np.zeros_like(current, dtype=complex)
    for j in range(current.shape[1]):
        spec_recon[:, j] = fft(current[:, j])

    spec_recon = fftshift(spec_recon).real
    spec_recon = spec_recon / np.abs(spec_recon).max()

    return spec_recon


# ═══════════════════════════════════════════════════════════
# 5. Metrics
# ═══════════════════════════════════════════════════════════
def compute_metrics(spec_gt, spec_recon):
    """Compute spectrum reconstruction quality metrics."""
    # Normalise both
    gt = spec_gt / np.abs(spec_gt).max()
    rec = spec_recon / np.abs(spec_recon).max()

    # PSNR
    data_range = gt.max() - gt.min()
    mse = np.mean((gt - rec) ** 2)
    psnr = float(10 * np.log10(data_range ** 2 / max(mse, 1e-30)))

    # SSIM
    ssim_val = float(ssim_fn(gt, rec, data_range=data_range))

    # CC
    cc = float(np.corrcoef(gt.ravel(), rec.ravel())[0, 1])

    # Relative error
    re = float(np.linalg.norm(gt - rec) / max(np.linalg.norm(gt), 1e-12))

    # RMSE
    rmse = float(np.sqrt(mse))

    # Peak detection accuracy (find peaks above threshold)
    from scipy.ndimage import label
    gt_mask = gt > 0.15 * gt.max()
    rec_mask = rec > 0.15 * rec.max()
    gt_labels, n_gt = label(gt_mask)
    rec_labels, n_rec = label(rec_mask)

    metrics = {
        "PSNR": psnr,
        "SSIM": ssim_val,
        "CC": cc,
        "RE": re,
        "RMSE": rmse,
        "n_peaks_gt": int(n_gt),
        "n_peaks_recon": int(n_rec),
    }
    return metrics


# ═══════════════════════════════════════════════════════════
# 6. Visualization
# ═══════════════════════════════════════════════════════════
def visualize_results(spec_gt, spec_recon, fid_nus, schedule,
                      metrics, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    vmax = np.percentile(np.abs(spec_gt), 99)

    # (a) Ground truth spectrum
    ax = axes[0, 0]
    ax.contourf(spec_gt.T, levels=30, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f'(a) Ground Truth ({N_PEAKS} peaks)')
    ax.set_xlabel('F1 [pts]')
    ax.set_ylabel('F2 [pts]')

    # (b) Reconstructed spectrum
    ax = axes[0, 1]
    ax.contourf(spec_recon.T, levels=30, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f'(b) IST Reconstruction (NUS {NUS_FRAC*100:.0f}%)')
    ax.set_xlabel('F1 [pts]')
    ax.set_ylabel('F2 [pts]')

    # (c) NUS schedule
    ax = axes[1, 0]
    ax.stem(np.where(schedule)[0], np.ones(schedule.sum()),
            linefmt='b-', markerfmt='b.', basefmt='k-')
    ax.set_xlim(0, N_F1)
    ax.set_xlabel('Indirect dimension index')
    ax.set_ylabel('Sampled')
    ax.set_title(f'(c) NUS Schedule ({schedule.sum()}/{N_F1})')

    # (d) 1D slice comparison
    ax = axes[1, 1]
    mid = spec_gt.shape[1] // 2
    ax.plot(spec_gt[:, mid], 'b-', lw=1.5, label='GT', alpha=0.8)
    ax.plot(spec_recon[:, mid], 'r--', lw=1.5, label='IST recon', alpha=0.8)
    ax.set_xlabel('F1 [pts]')
    ax.set_ylabel('Intensity')
    ax.set_title('(d) 1D Slice (F2 midpoint)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"nmrglue — 2D NMR NUS Reconstruction (IST)\n"
        f"PSNR={metrics['PSNR']:.1f} dB  |  SSIM={metrics['SSIM']:.4f}  |  "
        f"CC={metrics['CC']:.4f}  |  RE={metrics['RE']:.4f}",
        fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")


# ═══════════════════════════════════════════════════════════
# 7. Main Pipeline
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  nmrglue — 2D NMR NUS Reconstruction")
    print("=" * 65)

    fid_nus, spec_gt, fid_full, schedule = load_or_generate_data()

    print("\n[RECON] Running IST reconstruction ...")
    spec_recon = reconstruct(fid_nus, schedule)

    print("\n[EVAL] Computing metrics ...")
    metrics = compute_metrics(spec_gt, spec_recon)
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), spec_recon)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), spec_gt)

    visualize_results(spec_gt, spec_recon, fid_nus, schedule,
                      metrics, os.path.join(RESULTS_DIR, "reconstruction_result.png"))

    print("\n" + "=" * 65)
    print("  DONE — nmrglue NMR NUS reconstruction benchmark complete")
    print("=" * 65)
