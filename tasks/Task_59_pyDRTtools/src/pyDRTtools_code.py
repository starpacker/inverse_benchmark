"""
pyDRTtools — Distribution of Relaxation Times (DRT) Inversion
===============================================================
Task: Invert Electrochemical Impedance Spectroscopy (EIS) data to
      obtain the Distribution of Relaxation Times (DRT).

Inverse Problem:
    Given Z(ω) impedance spectrum, recover γ(τ) the distribution of
    relaxation times via Fredholm integral inversion:
      Z(ω) = R_∞ + R_pol ∫ γ(τ) / (1 + iωτ) d(ln τ)

Forward Model:
    Discretized Fredholm integral using pyDRTtools' basis functions.

Inverse Solver:
    Tikhonov regularization with pyDRTtools + cross-validation for λ.

Repo: https://github.com/ciuccislab/pyDRTtools
Paper: Wan et al. (2015), Electrochimica Acta, 184, 483–499.

Usage:
    /data/yjh/spectro_env/bin/python pyDRTtools_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.optimize import minimize
from skimage.metrics import structural_similarity as ssim_fn

# ── pyDRTtools library import ──────────────────────────────
try:
    import pyDRTtools
    from pyDRTtools.runs import simple_run
    from pyDRTtools.basics import EIS_object
    HAS_PYDRTT = True
except ImportError:
    HAS_PYDRTT = False
    print("[WARN] pyDRTtools not found, using Tikhonov fallback")

# ═══════════════════════════════════════════════════════════
# 1. Configuration
# ═══════════════════════════════════════════════════════════
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Frequency range
N_FREQ = 81
FREQ_MIN = 1e-2   # Hz
FREQ_MAX = 1e6    # Hz

# Ground truth DRT: sum of Gaussians in log(τ) space
N_TAU = 200
TAU_MIN = 1e-7
TAU_MAX = 1e2

# DRT peaks (3 distinct processes)
GT_PEAKS = [
    {"tau_center": 1e-5, "gamma_max": 0.15, "sigma": 0.5},
    {"tau_center": 1e-2, "gamma_max": 0.25, "sigma": 0.8},
    {"tau_center": 1e0,  "gamma_max": 0.10, "sigma": 0.6},
]
R_INF = 5.0     # Ohm (high-frequency resistance)
R_POL = 20.0    # Ohm (total polarisation resistance)

NOISE_LEVEL = 0.005  # relative noise on Z
SEED = 42


# ═══════════════════════════════════════════════════════════
# 2. Forward Model (Fredholm Integral)
# ═══════════════════════════════════════════════════════════
def generate_gt_drt(tau):
    """Generate ground truth γ(τ) as sum of Gaussians in log(τ)."""
    ln_tau = np.log(tau)
    gamma = np.zeros_like(tau)
    for p in GT_PEAKS:
        ln_center = np.log(p["tau_center"])
        gamma += p["gamma_max"] * np.exp(
            -(ln_tau - ln_center) ** 2 / (2 * p["sigma"] ** 2)
        )
    # Normalize so integral ~ 1
    d_ln_tau = np.diff(ln_tau)
    integral = np.sum(0.5 * (gamma[:-1] + gamma[1:]) * d_ln_tau)
    if integral > 0:
        gamma = gamma / integral
    return gamma


def forward_operator(gamma, tau, freq, R_inf, R_pol):
    """
    Compute EIS impedance from DRT via Fredholm integral.

    Z(ω) = R_∞ + R_pol ∫ γ(τ)/(1 + iωτ) d(ln τ)

    Parameters
    ----------
    gamma : np.ndarray  DRT values γ(τ).
    tau : np.ndarray    Relaxation times [s].
    freq : np.ndarray   Frequencies [Hz].
    R_inf : float       High-frequency resistance [Ω].
    R_pol : float       Polarisation resistance [Ω].

    Returns
    -------
    Z : np.ndarray      Complex impedance [Ω].
    """
    omega = 2 * np.pi * freq
    ln_tau = np.log(tau)
    d_ln_tau = np.zeros_like(ln_tau)
    d_ln_tau[1:-1] = (ln_tau[2:] - ln_tau[:-2]) / 2
    d_ln_tau[0] = ln_tau[1] - ln_tau[0]
    d_ln_tau[-1] = ln_tau[-1] - ln_tau[-2]

    Z = np.full(len(freq), R_inf, dtype=complex)
    for i, w in enumerate(omega):
        integrand = gamma / (1 + 1j * w * tau)
        Z[i] += R_pol * np.sum(integrand * d_ln_tau)

    return Z


# ═══════════════════════════════════════════════════════════
# 3. Data Generation
# ═══════════════════════════════════════════════════════════
def load_or_generate_data():
    """Generate synthetic EIS data from known DRT."""
    print("[DATA] Generating synthetic EIS from DRT ...")

    freq = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_FREQ)
    tau = np.logspace(np.log10(TAU_MIN), np.log10(TAU_MAX), N_TAU)

    gamma_gt = generate_gt_drt(tau)

    Z_clean = forward_operator(gamma_gt, tau, freq, R_INF, R_POL)

    # Add noise
    rng = np.random.default_rng(SEED)
    Z_mag = np.abs(Z_clean)
    noise = NOISE_LEVEL * Z_mag * (
        rng.standard_normal(N_FREQ) + 1j * rng.standard_normal(N_FREQ)
    )
    Z_noisy = Z_clean + noise

    print(f"[DATA] {N_FREQ} frequencies: [{FREQ_MIN:.0e}, {FREQ_MAX:.0e}] Hz")
    print(f"[DATA] |Z| range: [{Z_mag.min():.2f}, {Z_mag.max():.2f}] Ω")
    print(f"[DATA] DRT peaks at τ = {[p['tau_center'] for p in GT_PEAKS]}")

    return freq, Z_clean, Z_noisy, tau, gamma_gt


# ═══════════════════════════════════════════════════════════
# 4. Inverse Solver
# ═══════════════════════════════════════════════════════════
def reconstruct(freq, Z_noisy, tau):
    """
    Recover DRT from noisy EIS data.

    Uses pyDRTtools if available (Tikhonov + GCV/L-curve),
    otherwise falls back to direct Tikhonov with scipy.

    Parameters
    ----------
    freq : np.ndarray   Frequencies [Hz].
    Z_noisy : np.ndarray Complex impedance data.
    tau : np.ndarray     Relaxation time grid.

    Returns
    -------
    gamma_rec : np.ndarray  Recovered DRT.
    Z_fit : np.ndarray      Fitted impedance.
    """
    if HAS_PYDRTT:
        print("[RECON] Using pyDRTtools Tikhonov inversion ...")
        try:
            # Create EIS object for pyDRTtools
            eis = EIS_object(freq, Z_noisy.real, Z_noisy.imag)
            # Run DRT analysis
            result = simple_run(eis, rbf_type='Gaussian',
                               data_used='Combined Re-Im',
                               induct_used=0, der_used='1st',
                               lambda_value=1e-3,
                               NMC_sample=0)
            gamma_rec = result.gamma
            tau_out = result.tau

            # Interpolate to our tau grid
            gamma_rec_interp = np.interp(np.log10(tau), np.log10(tau_out), gamma_rec)
            gamma_rec_interp = np.maximum(gamma_rec_interp, 0)

            # Compute fit
            Z_fit = forward_operator(gamma_rec_interp, tau, freq, R_INF, R_POL)
            return gamma_rec_interp, Z_fit
        except Exception as e:
            print(f"[WARN] pyDRTtools failed: {e}, using fallback")

    # Fallback: Direct Tikhonov
    return _reconstruct_tikhonov(freq, Z_noisy, tau)


def _reconstruct_tikhonov(freq, Z_noisy, tau):
    """Tikhonov regularisation for DRT inversion."""
    print("[RECON] Tikhonov DRT inversion (Fredholm integral) ...")

    omega = 2 * np.pi * freq
    ln_tau = np.log(tau)
    d_ln_tau = np.zeros_like(ln_tau)
    d_ln_tau[1:-1] = (ln_tau[2:] - ln_tau[:-2]) / 2
    d_ln_tau[0] = ln_tau[1] - ln_tau[0]
    d_ln_tau[-1] = ln_tau[-1] - ln_tau[-2]

    # Build kernel matrix A where Z = R_inf + R_pol * A @ gamma
    A = np.zeros((N_FREQ, N_TAU), dtype=complex)
    for i, w in enumerate(omega):
        A[i, :] = R_POL * d_ln_tau / (1 + 1j * w * tau)

    # Stack real and imaginary parts
    A_stack = np.vstack([A.real, A.imag])
    b = np.hstack([
        Z_noisy.real - R_INF,
        Z_noisy.imag,
    ])

    # Smoothness matrix
    D = np.zeros((N_TAU - 1, N_TAU))
    for i in range(N_TAU - 1):
        D[i, i] = -1
        D[i, i + 1] = 1

    # GCV for lambda selection
    lambdas = np.logspace(-6, 0, 20)
    gcv_scores = []
    for lam in lambdas:
        AtA = A_stack.T @ A_stack + lam * D.T @ D
        try:
            gamma = np.linalg.solve(AtA, A_stack.T @ b)
        except np.linalg.LinAlgError:
            gcv_scores.append(1e20)
            continue
        gamma = np.maximum(gamma, 0)
        resid = b - A_stack @ gamma
        n = len(b)
        try:
            H = A_stack @ np.linalg.solve(AtA, A_stack.T)
            trace_H = np.trace(H)
        except Exception:
            trace_H = N_TAU
        gcv = (np.sum(resid ** 2) / n) / max((1 - trace_H / n) ** 2, 1e-12)
        gcv_scores.append(gcv)

    best_lam = lambdas[np.argmin(gcv_scores)]
    print(f"[RECON]   Best λ = {best_lam:.2e} (GCV)")

    AtA = A_stack.T @ A_stack + best_lam * D.T @ D
    gamma_rec = np.linalg.solve(AtA, A_stack.T @ b)
    gamma_rec = np.maximum(gamma_rec, 0)

    Z_fit = forward_operator(gamma_rec, tau, freq, R_INF, R_POL)
    return gamma_rec, Z_fit


# ═══════════════════════════════════════════════════════════
# 5. Metrics
# ═══════════════════════════════════════════════════════════
def compute_metrics(gamma_gt, gamma_rec, Z_clean, Z_fit, tau):
    """Compute DRT reconstruction metrics."""
    # DRT metrics (normalized)
    g_gt = gamma_gt / max(gamma_gt.max(), 1e-12)
    g_rec = gamma_rec / max(gamma_rec.max(), 1e-12)

    cc_drt = float(np.corrcoef(g_gt, g_rec)[0, 1])
    re_drt = float(np.linalg.norm(g_gt - g_rec) / max(np.linalg.norm(g_gt), 1e-12))
    rmse_drt = float(np.sqrt(np.mean((g_gt - g_rec) ** 2)))

    data_range = g_gt.max() - g_gt.min()
    mse = np.mean((g_gt - g_rec) ** 2)
    psnr_drt = float(10 * np.log10(data_range ** 2 / max(mse, 1e-30)))
    tile_rows = 7
    a2d = np.tile(g_gt, (tile_rows, 1))
    b2d = np.tile(g_rec, (tile_rows, 1))
    ssim_drt = float(ssim_fn(a2d, b2d,
                              data_range=data_range, win_size=7))

    # Impedance fit metrics
    Z_resid = Z_clean - Z_fit
    rmse_Z = float(np.sqrt(np.mean(np.abs(Z_resid) ** 2)))
    cc_Z_re = float(np.corrcoef(Z_clean.real, Z_fit.real)[0, 1])
    cc_Z_im = float(np.corrcoef(Z_clean.imag, Z_fit.imag)[0, 1])

    # Peak detection
    from scipy.signal import find_peaks
    peaks_gt, _ = find_peaks(g_gt, height=0.1)
    peaks_rec, _ = find_peaks(g_rec, height=0.1)

    metrics = {
        "PSNR_DRT": psnr_drt,
        "SSIM_DRT": ssim_drt,
        "CC_DRT": cc_drt,
        "RE_DRT": re_drt,
        "RMSE_DRT": rmse_drt,
        "CC_Z_real": cc_Z_re,
        "CC_Z_imag": cc_Z_im,
        "RMSE_Z": rmse_Z,
        "n_peaks_gt": len(peaks_gt),
        "n_peaks_rec": len(peaks_rec),
    }
    return metrics


# ═══════════════════════════════════════════════════════════
# 6. Visualization
# ═══════════════════════════════════════════════════════════
def visualize_results(freq, Z_clean, Z_noisy, Z_fit,
                      tau, gamma_gt, gamma_rec, metrics, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) DRT
    ax = axes[0, 0]
    ax.semilogx(tau, gamma_gt / max(gamma_gt.max(), 1e-12),
                'b-', lw=2, label='GT')
    ax.semilogx(tau, gamma_rec / max(gamma_rec.max(), 1e-12),
                'r--', lw=2, label='Recovered')
    ax.set_xlabel('τ [s]')
    ax.set_ylabel('γ(τ) [normalised]')
    ax.set_title('(a) Distribution of Relaxation Times')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Nyquist plot
    ax = axes[0, 1]
    ax.plot(Z_clean.real, -Z_clean.imag, 'b-', lw=2, label='GT')
    ax.plot(Z_noisy.real, -Z_noisy.imag, 'k.', ms=3, alpha=0.5, label='Noisy')
    ax.plot(Z_fit.real, -Z_fit.imag, 'r--', lw=2, label='Fit')
    ax.set_xlabel("Z' [Ω]")
    ax.set_ylabel("-Z'' [Ω]")
    ax.set_title('(b) Nyquist Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # (c) Bode magnitude
    ax = axes[1, 0]
    ax.loglog(freq, np.abs(Z_clean), 'b-', lw=2, label='GT')
    ax.loglog(freq, np.abs(Z_noisy), 'k.', ms=3, alpha=0.5, label='Noisy')
    ax.loglog(freq, np.abs(Z_fit), 'r--', lw=2, label='Fit')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('|Z| [Ω]')
    ax.set_title('(c) Bode Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # (d) Bode phase
    ax = axes[1, 1]
    ax.semilogx(freq, np.degrees(np.angle(Z_clean)), 'b-', lw=2, label='GT')
    ax.semilogx(freq, np.degrees(np.angle(Z_noisy)), 'k.', ms=3, alpha=0.5)
    ax.semilogx(freq, np.degrees(np.angle(Z_fit)), 'r--', lw=2, label='Fit')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Phase [°]')
    ax.set_title('(d) Bode Phase')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"pyDRTtools — DRT Inversion from EIS\n"
        f"PSNR={metrics['PSNR_DRT']:.1f} dB  |  "
        f"SSIM={metrics['SSIM_DRT']:.4f}  |  CC={metrics['CC_DRT']:.4f}",
        fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved → {save_path}")


# ═══════════════════════════════════════════════════════════
# 7. Main
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  pyDRTtools — DRT Inversion from EIS")
    print("=" * 65)

    freq, Z_clean, Z_noisy, tau, gamma_gt = load_or_generate_data()

    print("\n[RECON] Inverting EIS → DRT ...")
    gamma_rec, Z_fit = reconstruct(freq, Z_noisy, tau)

    print("\n[EVAL] Computing metrics ...")
    metrics = compute_metrics(gamma_gt, gamma_rec, Z_clean, Z_fit, tau)
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), gamma_rec)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), gamma_gt)

    visualize_results(freq, Z_clean, Z_noisy, Z_fit,
                      tau, gamma_gt, gamma_rec, metrics,
                      os.path.join(RESULTS_DIR, "reconstruction_result.png"))

    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65)
