"""
CDI — Coherent Diffraction Imaging Phase Retrieval
=====================================================
Task #73: Reconstruct complex-valued object from its far-field
          diffraction intensity pattern using iterative phase retrieval.

Inverse Problem:
    Given |F{ρ(r)}|² (diffraction intensity), recover ρ(r) (object).
    The phase information is lost — this is the "phase problem".

Forward Model:
    Far-field diffraction (Fraunhofer):
    I(q) = |F{ρ(r)}|² = |Σ_r ρ(r) exp(-i q·r)|²
    Only intensity is measured; phase is lost.

Inverse Solver:
    1) HIO (Hybrid Input-Output) — Fienup's algorithm
    2) ER (Error Reduction) — Gerchberg-Saxton
    3) HIO+ER alternating with shrink-wrap support update

Repo: https://github.com/mcherukara/CDI_image_reconstruction
Paper: Fienup (1982), Applied Optics; Marchesini (2003), Rev. Sci. Instr.

Usage: /data/yjh/spectro_env/bin/python CDI_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage.metrics import structural_similarity as ssim_fn

# ─── Configuration ─────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OBJ_SIZE = 64               # Object size (64×64)
DET_SIZE = 128               # Detector size (oversampled)
N_HIO = 600                  # HIO iterations
N_ER = 300                    # ER iterations
BETA_HIO = 0.9               # HIO feedback parameter
SHRINKWRAP_INTERVAL = 50     # Shrink-wrap support update interval
SHRINKWRAP_SIGMA = 2.5       # Gaussian blur for support estimation
SHRINKWRAP_THRESHOLD = 0.08  # Threshold for support
NOISE_SNR_DB = 45            # Photon noise level
N_STARTS = 3                 # Multi-start attempts
SEED = 42


# ─── Data Generation ──────────────────────────────────────────────
def generate_object(obj_size, det_size):
    """
    Create a complex-valued object with amplitude and phase structure.
    Represents a nanocrystal or biological sample.
    """
    # Amplitude: geometric features
    amp = np.zeros((det_size, det_size))
    cx, cy = det_size // 2, det_size // 2
    Y, X = np.mgrid[:det_size, :det_size]

    # Crystal-like shape
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    amp[r < obj_size // 3] = 0.8

    # Internal structure
    amp[(np.abs(X - cx) < obj_size // 6) &
        (np.abs(Y - cy) < obj_size // 4)] = 1.0

    # Small features
    amp[(X - cx + 10)**2 + (Y - cy + 8)**2 < 16] = 0.6
    amp[(X - cx - 8)**2 + (Y - cy - 12)**2 < 9] = 0.7

    # Phase: smooth strain field
    phase = np.zeros((det_size, det_size))
    phase = 0.5 * np.sin(2 * np.pi * (X - cx) / obj_size) * (amp > 0)
    phase += 0.3 * np.cos(2 * np.pi * (Y - cy) / (obj_size * 0.8)) * (amp > 0)

    obj = amp * np.exp(1j * phase)
    support = amp > 0.01

    return obj, support


def forward_diffraction(obj, snr_db, rng):
    """
    Compute far-field diffraction intensity pattern.
    I = |FFT(obj)|²
    Add Poisson-like photon noise.
    """
    F_obj = fftshift(fft2(ifftshift(obj)))
    intensity_clean = np.abs(F_obj)**2

    # Normalise
    intensity_clean /= intensity_clean.max()

    # Poisson-like noise
    sig_power = np.mean(intensity_clean**2)
    noise_power = sig_power / (10**(snr_db / 10))
    noise = np.sqrt(noise_power) * rng.standard_normal(intensity_clean.shape)
    intensity_noisy = np.maximum(intensity_clean + noise, 0)

    return intensity_clean, intensity_noisy, F_obj


# ─── Inverse Solver: HIO ──────────────────────────────────────────
def hio_iteration(obj_est, sqrt_intensity, support, beta):
    """
    Single HIO (Hybrid Input-Output) iteration.

    1. FFT → replace amplitude with measured √I → IFFT
    2. Apply support constraint with HIO feedback
    """
    # Fourier constraint: replace amplitude
    F_est = fftshift(fft2(ifftshift(obj_est)))
    phase_est = np.angle(F_est)
    F_constrained = sqrt_intensity * np.exp(1j * phase_est)
    obj_new = fftshift(ifft2(ifftshift(F_constrained)))

    # Real-space constraint (HIO)
    obj_out = obj_est.copy()
    outside_support = ~support

    # Inside support: keep new estimate
    obj_out[support] = obj_new[support]
    # Outside support: HIO feedback
    obj_out[outside_support] = obj_est[outside_support] - beta * obj_new[outside_support]

    return obj_out


def er_iteration(obj_est, sqrt_intensity, support):
    """
    Single ER (Error Reduction) iteration.
    Same as HIO but with hard support projection.
    """
    F_est = fftshift(fft2(ifftshift(obj_est)))
    phase_est = np.angle(F_est)
    F_constrained = sqrt_intensity * np.exp(1j * phase_est)
    obj_new = fftshift(ifft2(ifftshift(F_constrained)))

    # Hard support constraint
    obj_new[~support] = 0
    return obj_new


def update_support_shrinkwrap(obj_est, sigma, threshold):
    """
    Shrink-wrap support estimation: blur amplitude, threshold.
    """
    amp = np.abs(obj_est)
    amp_smooth = gaussian_filter(amp, sigma=sigma)
    amp_smooth /= max(amp_smooth.max(), 1e-12)
    support = amp_smooth > threshold
    # Dilate slightly for stability
    support = binary_dilation(support, iterations=1)
    return support


def phase_retrieval(intensity_noisy, support_init, n_hio, n_er, beta,
                     shrinkwrap_interval, shrinkwrap_sigma, shrinkwrap_threshold,
                     rng):
    """
    Full HIO+ER phase retrieval with shrink-wrap support.
    """
    sqrt_intensity = np.sqrt(intensity_noisy)
    det_size = intensity_noisy.shape[0]

    # Random initial guess
    phase_init = 2 * np.pi * rng.random((det_size, det_size))
    obj_est = support_init.astype(complex) * np.exp(1j * phase_init)

    support = support_init.copy()
    errors = []

    print(f"  Phase retrieval: {n_hio} HIO + {n_er} ER iterations")

    # HIO phase
    for it in range(n_hio):
        obj_est = hio_iteration(obj_est, sqrt_intensity, support, beta)

        # Shrink-wrap support update
        if (it + 1) % shrinkwrap_interval == 0:
            support = update_support_shrinkwrap(
                obj_est, shrinkwrap_sigma, shrinkwrap_threshold
            )
            print(f"    HIO iter {it+1:4d}: support pixels = {support.sum()}")

        # Error metric
        F_est = fftshift(fft2(ifftshift(obj_est)))
        err = np.sqrt(np.mean((np.abs(F_est) - sqrt_intensity)**2))
        errors.append(err)

        if (it + 1) % 50 == 0:
            print(f"    HIO iter {it+1:4d}: R-factor = {err:.6f}")

    # ER phase (refinement)
    for it in range(n_er):
        obj_est = er_iteration(obj_est, sqrt_intensity, support)

        if (it + 1) % 25 == 0:
            F_est = fftshift(fft2(ifftshift(obj_est)))
            err = np.sqrt(np.mean((np.abs(F_est) - sqrt_intensity)**2))
            errors.append(err)
            print(f"    ER  iter {it+1:4d}: R-factor = {err:.6f}")

    return obj_est, support, errors


# ─── Phase Alignment ──────────────────────────────────────────────
def align_phase(obj_gt, obj_rec):
    """
    Remove global phase ambiguity and possible twin-image flip.
    """
    # Try direct and conjugate
    candidates = [obj_rec, np.conj(obj_rec), np.flip(obj_rec),
                  np.conj(np.flip(obj_rec))]

    best_cc = -1
    best = obj_rec

    for cand in candidates:
        # Find optimal global phase
        cross = np.sum(obj_gt * np.conj(cand))
        phi = np.angle(cross)
        cand_aligned = cand * np.exp(1j * phi)

        cc = np.abs(np.corrcoef(
            np.abs(obj_gt).ravel(), np.abs(cand_aligned).ravel()
        )[0, 1])
        if cc > best_cc:
            best_cc = cc
            best = cand_aligned

    return best


# ─── Metrics ───────────────────────────────────────────────────────
def compute_metrics(obj_gt, obj_rec):
    """Compute reconstruction metrics on amplitude."""
    amp_gt = np.abs(obj_gt)
    amp_rec = np.abs(obj_rec)

    amp_gt_n = amp_gt / max(amp_gt.max(), 1e-12)
    amp_rec_n = amp_rec / max(amp_rec.max(), 1e-12)

    data_range = 1.0
    mse = np.mean((amp_gt_n - amp_rec_n)**2)
    psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(amp_gt_n, amp_rec_n, data_range=data_range))
    cc = float(np.corrcoef(amp_gt_n.ravel(), amp_rec_n.ravel())[0, 1])
    re = float(np.linalg.norm(amp_gt_n - amp_rec_n) /
               max(np.linalg.norm(amp_gt_n), 1e-12))
    rmse = float(np.sqrt(mse))

    # Phase error (inside support only)
    support = amp_gt > 0.01 * amp_gt.max()
    if support.sum() > 0:
        phase_gt = np.angle(obj_gt[support])
        phase_rec = np.angle(obj_rec[support])
        phase_err = np.angle(np.exp(1j * (phase_gt - phase_rec)))
        phase_rmse = float(np.sqrt(np.mean(phase_err**2)))
    else:
        phase_rmse = np.pi

    return {
        "PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse,
        "phase_RMSE_rad": phase_rmse,
    }


# ─── Visualization ─────────────────────────────────────────────────
def visualize_results(obj_gt, obj_rec, intensity, errors, metrics, save_path):
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    # Diffraction pattern
    axes[0, 0].imshow(np.log10(intensity + 1e-6), cmap='viridis')
    axes[0, 0].set_title('Diffraction Pattern (log)')

    # GT amplitude
    axes[0, 1].imshow(np.abs(obj_gt), cmap='gray')
    axes[0, 1].set_title('GT Amplitude')

    # Recon amplitude
    axes[0, 2].imshow(np.abs(obj_rec), cmap='gray')
    axes[0, 2].set_title('Recon Amplitude')

    # Amplitude error
    err_amp = np.abs(np.abs(obj_gt) - np.abs(obj_rec))
    axes[0, 3].imshow(err_amp, cmap='hot')
    axes[0, 3].set_title('|Amplitude Error|')

    # GT phase
    support = np.abs(obj_gt) > 0.01 * np.abs(obj_gt).max()
    phase_gt = np.angle(obj_gt) * support
    axes[1, 0].imshow(phase_gt, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1, 0].set_title('GT Phase')

    # Recon phase
    phase_rec = np.angle(obj_rec) * support
    axes[1, 1].imshow(phase_rec, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1, 1].set_title('Recon Phase')

    # Phase error
    phase_err = np.angle(np.exp(1j * (phase_gt - phase_rec))) * support
    axes[1, 2].imshow(phase_err, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 2].set_title('Phase Error')

    # Convergence
    if errors:
        axes[1, 3].semilogy(errors)
        axes[1, 3].set_title('Convergence (R-factor)')
        axes[1, 3].set_xlabel('Iteration')
        axes[1, 3].grid(True)

    for row in axes:
        for ax in row:
            ax.axis('off') if ax != axes[1, 3] else None

    fig.suptitle(
        f"CDI — Phase Retrieval (HIO+ER)\n"
        f"PSNR={metrics['PSNR']:.1f} dB | CC={metrics['CC']:.4f} | "
        f"Phase RMSE={metrics['phase_RMSE_rad']:.3f} rad",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─── Main Pipeline ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  CDI — Coherent Diffraction Imaging Phase Retrieval")
    print("=" * 70)

    rng = np.random.default_rng(SEED)

    # Stage 1: Data Generation
    print("\n[STAGE 1] Data Generation")
    obj_gt, support_gt = generate_object(OBJ_SIZE, DET_SIZE)
    print(f"  Object: {obj_gt.shape} (complex)")
    print(f"  Support pixels: {support_gt.sum()}")
    print(f"  Oversampling: {DET_SIZE/OBJ_SIZE:.1f}×")

    # Stage 2: Forward — Diffraction
    print("\n[STAGE 2] Forward — Far-Field Diffraction I = |FFT(ρ)|²")
    intensity_clean, intensity_noisy, F_gt = forward_diffraction(
        obj_gt, NOISE_SNR_DB, rng
    )
    print(f"  Intensity range: [{intensity_noisy.min():.4f}, "
          f"{intensity_noisy.max():.4f}]")

    # Stage 3: Inverse — Phase Retrieval (multi-start)
    print(f"\n[STAGE 3] Inverse — HIO + ER Phase Retrieval ({N_STARTS} starts)")
    best_obj_rec = None
    best_err = float('inf')
    best_errors = []
    best_support = None
    for start_i in range(N_STARTS):
        rng_start = np.random.default_rng(SEED + start_i)
        obj_rec_i, support_i, errors_i = phase_retrieval(
            intensity_noisy, support_gt, N_HIO, N_ER, BETA_HIO,
            SHRINKWRAP_INTERVAL, SHRINKWRAP_SIGMA, SHRINKWRAP_THRESHOLD, rng_start
        )
        final_err = errors_i[-1] if errors_i else float('inf')
        print(f"  Start {start_i+1}/{N_STARTS}: final R-factor = {final_err:.6f}")
        if final_err < best_err:
            best_err = final_err
            best_obj_rec = obj_rec_i
            best_errors = errors_i
            best_support = support_i
    obj_rec = best_obj_rec
    support_final = best_support
    errors = best_errors
    print(f"  Best start: R-factor = {best_err:.6f}")

    # Align phase
    obj_rec = align_phase(obj_gt, obj_rec)

    # Stage 4: Evaluation
    print("\n[STAGE 4] Evaluation Metrics:")
    metrics = compute_metrics(obj_gt, obj_rec)
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")

    # Standard metrics
    std_metrics = {k: v for k, v in metrics.items()
                   if k in ["PSNR", "SSIM", "CC", "RE", "RMSE"]}
    std_metrics["phase_RMSE_rad"] = metrics["phase_RMSE_rad"]

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(std_metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), np.abs(obj_rec))
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), np.abs(obj_gt))

    visualize_results(obj_gt, obj_rec, intensity_noisy, errors, metrics,
                      os.path.join(RESULTS_DIR, "reconstruction_result.png"))

    print("\n" + "=" * 70)
    print("  DONE — Results saved to", RESULTS_DIR)
    print("=" * 70)
