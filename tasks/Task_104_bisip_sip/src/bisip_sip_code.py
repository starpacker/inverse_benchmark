"""
bisip_sip - Spectral Induced Polarization: Cole-Cole Model Inversion
=====================================================================
From complex resistivity spectra ρ*(ω), fit Cole-Cole model parameters
(ρ_0, m, τ, c) using nonlinear least squares.

Physics:
  - Forward: Cole-Cole model ρ*(ω) = ρ_0 [1 - m(1 - 1/(1+(iωτ)^c))]
  - Inverse: Scipy curve_fit for parameter estimation
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json
import time
from scipy.optimize import minimize
from skimage.metrics import structural_similarity as ssim

# ── paths ──────────────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
ASSETS_DIR  = "/data/yjh/website_assets/Task_104_bisip_sip"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── parameters ─────────────────────────────────────────────────────
N_FREQ         = 30          # number of frequency points
FREQ_MIN       = 0.001       # Hz
FREQ_MAX       = 1000.0      # Hz
NOISE_LEVEL    = 0.02        # 2% relative noise
SEED           = 42
N_SPECTRA      = 5           # number of different spectra to fit

np.random.seed(SEED)

# Ground truth Cole-Cole parameters for multiple spectra
GT_PARAMS = [
    {"rho0": 100.0, "m": 0.3, "tau": 0.1,  "c": 0.5},
    {"rho0": 250.0, "m": 0.5, "tau": 1.0,  "c": 0.7},
    {"rho0":  50.0, "m": 0.2, "tau": 0.01, "c": 0.3},
    {"rho0": 500.0, "m": 0.4, "tau": 10.0, "c": 0.6},
    {"rho0": 150.0, "m": 0.6, "tau": 0.5,  "c": 0.8},
]


# ====================================================================
# 1. Cole-Cole forward model
# ====================================================================
def cole_cole(freq, rho0, m, tau, c):
    """
    Cole-Cole complex resistivity model.
    ρ*(ω) = ρ_0 × [1 - m × (1 - 1/(1 + (iωτ)^c))]
    
    Parameters:
        freq: frequency array (Hz)
        rho0: DC resistivity (Ohm·m)
        m:    chargeability (0-1)
        tau:  time constant (s)
        c:    frequency exponent (0-1)
    
    Returns:
        Complex resistivity array
    """
    omega = 2.0 * np.pi * freq
    z = (1j * omega * tau) ** c
    rho_star = rho0 * (1.0 - m * (1.0 - 1.0 / (1.0 + z)))
    return rho_star


def cole_cole_amplitude_phase(freq, rho0, m, tau, c):
    """Return amplitude and phase of Cole-Cole model."""
    rho_star = cole_cole(freq, rho0, m, tau, c)
    amplitude = np.abs(rho_star)
    phase = -np.angle(rho_star, deg=True)  # phase in mrad (negative convention)
    phase_mrad = -np.angle(rho_star) * 1000.0  # mrad
    return amplitude, phase_mrad


# ====================================================================
# 2. Generate synthetic data
# ====================================================================
def generate_data(freq, params, noise_level):
    """Generate noisy complex resistivity data from Cole-Cole model."""
    rho_true = cole_cole(freq, params["rho0"], params["m"],
                         params["tau"], params["c"])

    # Add relative noise to real and imaginary parts
    noise_re = noise_level * np.abs(rho_true) * np.random.randn(len(freq))
    noise_im = noise_level * np.abs(rho_true) * np.random.randn(len(freq))
    rho_noisy = rho_true + noise_re + 1j * noise_im

    return rho_noisy, rho_true


# ====================================================================
# 3. Inverse: Nonlinear least squares fitting
# ====================================================================
def objective(params_vec, freq, rho_obs):
    """
    Least-squares objective: minimize misfit between observed and modeled
    complex resistivity.
    """
    rho0, m, tau, c = params_vec
    # Enforce bounds implicitly
    if rho0 <= 0 or m <= 0 or m >= 1 or tau <= 0 or c <= 0 or c >= 1:
        return 1e10

    rho_model = cole_cole(freq, rho0, m, tau, c)

    # Normalized misfit (amplitude + phase)
    amp_obs = np.abs(rho_obs)
    amp_mod = np.abs(rho_model)
    phase_obs = np.angle(rho_obs)
    phase_mod = np.angle(rho_model)

    misfit_amp = np.sum(((amp_obs - amp_mod) / amp_obs)**2)
    misfit_phase = np.sum((phase_obs - phase_mod)**2)

    return misfit_amp + misfit_phase


def invert_cole_cole(freq, rho_obs, true_params=None):
    """
    Invert for Cole-Cole parameters using differential evolution-like
    multi-start optimization.
    """
    best_result = None
    best_cost = np.inf

    # Multi-start with different initial guesses
    rho0_guesses = [50.0, 150.0, 300.0, 500.0]
    m_guesses = [0.2, 0.4, 0.6]
    tau_guesses = [0.01, 0.1, 1.0, 10.0]
    c_guesses = [0.3, 0.5, 0.7]

    bounds = [(1.0, 1000.0), (0.01, 0.99), (1e-4, 100.0), (0.05, 0.95)]

    for rho0_init in rho0_guesses:
        for m_init in m_guesses:
            for tau_init in tau_guesses:
                for c_init in c_guesses:
                    x0 = [rho0_init, m_init, tau_init, c_init]
                    try:
                        result = minimize(
                            objective, x0, args=(freq, rho_obs),
                            method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 500, 'ftol': 1e-12}
                        )
                        if result.fun < best_cost:
                            best_cost = result.fun
                            best_result = result
                    except Exception:
                        continue

    if best_result is not None:
        rho0, m, tau, c = best_result.x
        return {"rho0": rho0, "m": m, "tau": tau, "c": c}
    else:
        return {"rho0": 100.0, "m": 0.3, "tau": 0.1, "c": 0.5}


# ====================================================================
# 4. Metrics
# ====================================================================
def compute_spectrum_metrics(rho_true, rho_fit):
    """Compute PSNR and CC for spectrum comparison."""
    # Amplitude comparison
    amp_true = np.abs(rho_true)
    amp_fit = np.abs(rho_fit)

    # Normalize
    amp_true_n = amp_true / amp_true.max()
    amp_fit_n = amp_fit / amp_fit.max()

    # PSNR
    mse = np.mean((amp_true_n - amp_fit_n)**2)
    psnr = 10.0 * np.log10(1.0 / mse) if mse > 1e-15 else 100.0

    # CC
    t_z = amp_true_n - amp_true_n.mean()
    f_z = amp_fit_n - amp_fit_n.mean()
    denom = np.sqrt(np.sum(t_z**2) * np.sum(f_z**2))
    cc = np.sum(t_z * f_z) / denom if denom > 1e-15 else 0.0

    return float(psnr), float(cc)


def compute_param_errors(gt_params, fit_params):
    """Compute relative errors for each Cole-Cole parameter."""
    errors = {}
    for key in ["rho0", "m", "tau", "c"]:
        rel_err = abs(fit_params[key] - gt_params[key]) / abs(gt_params[key]) * 100.0
        errors[key] = float(rel_err)
    return errors


# ====================================================================
# 5. Visualization
# ====================================================================
def plot_results(freq, all_results, avg_metrics):
    """Create Bode plots for all spectra and parameter comparisons."""
    n = len(all_results)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, res in enumerate(all_results):
        rho_true = res["rho_true"]
        rho_obs = res["rho_obs"]
        rho_fit = res["rho_fit"]

        amp_true = np.abs(rho_true)
        amp_obs = np.abs(rho_obs)
        amp_fit = np.abs(rho_fit)
        phase_true = -np.angle(rho_true) * 1000.0  # mrad
        phase_obs = -np.angle(rho_obs) * 1000.0
        phase_fit = -np.angle(rho_fit) * 1000.0

        # Amplitude plot
        ax = axes[i, 0]
        ax.semilogx(freq, amp_true, 'k-', lw=2, label='True')
        ax.semilogx(freq, amp_obs, 'b.', ms=4, alpha=0.5, label='Observed')
        ax.semilogx(freq, amp_fit, 'r--', lw=1.5, label='Fit')
        ax.set_ylabel("|ρ*| (Ω·m)")
        ax.set_title(f"Spectrum {i+1}: ρ₀={res['gt']['rho0']:.0f}, "
                     f"m={res['gt']['m']:.1f}, τ={res['gt']['tau']:.2f}, "
                     f"c={res['gt']['c']:.1f}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if i == n - 1:
            ax.set_xlabel("Frequency (Hz)")

        # Phase plot
        ax = axes[i, 1]
        ax.semilogx(freq, phase_true, 'k-', lw=2, label='True')
        ax.semilogx(freq, phase_obs, 'b.', ms=4, alpha=0.5, label='Observed')
        ax.semilogx(freq, phase_fit, 'r--', lw=1.5, label='Fit')
        ax.set_ylabel("-φ (mrad)")
        errs = res["param_errors"]
        ax.set_title(f"Errors: ρ₀={errs['rho0']:.1f}%, m={errs['m']:.1f}%, "
                     f"τ={errs['tau']:.1f}%, c={errs['c']:.1f}%", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if i == n - 1:
            ax.set_xlabel("Frequency (Hz)")

    plt.suptitle(f"Cole-Cole SIP Inversion — "
                 f"Avg PSNR={avg_metrics['PSNR']:.1f}dB, CC={avg_metrics['CC']:.3f}",
                 fontsize=14, y=1.01)
    plt.tight_layout()

    for d in [RESULTS_DIR, ASSETS_DIR]:
        fig.savefig(os.path.join(d, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
        fig.savefig(os.path.join(d, "vis_result.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ====================================================================
# 6. Main
# ====================================================================
def main():
    print("=" * 60)
    print("Task 104: Spectral Induced Polarization — Cole-Cole Inversion")
    print("=" * 60)

    # Frequency array
    freq = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_FREQ)
    print(f"[1] Frequencies: {N_FREQ} points, {FREQ_MIN}-{FREQ_MAX} Hz")

    all_results = []
    all_psnr = []
    all_cc = []

    for idx, gt_p in enumerate(GT_PARAMS):
        print(f"\n--- Spectrum {idx+1}/{N_SPECTRA} ---")
        print(f"    GT: ρ₀={gt_p['rho0']}, m={gt_p['m']}, τ={gt_p['tau']}, c={gt_p['c']}")

        # Forward model + noise
        rho_obs, rho_true = generate_data(freq, gt_p, NOISE_LEVEL)

        # Inverse
        t0 = time.time()
        fit_p = invert_cole_cole(freq, rho_obs, gt_p)
        elapsed = time.time() - t0
        print(f"    Fit: ρ₀={fit_p['rho0']:.2f}, m={fit_p['m']:.4f}, "
              f"τ={fit_p['tau']:.4f}, c={fit_p['c']:.4f}  ({elapsed:.1f}s)")

        # Compute fit spectrum
        rho_fit = cole_cole(freq, fit_p["rho0"], fit_p["m"], fit_p["tau"], fit_p["c"])

        # Metrics
        psnr, cc = compute_spectrum_metrics(rho_true, rho_fit)
        param_errors = compute_param_errors(gt_p, fit_p)
        print(f"    PSNR={psnr:.1f} dB, CC={cc:.4f}")
        print(f"    Param errors: {param_errors}")

        all_psnr.append(psnr)
        all_cc.append(cc)
        all_results.append({
            "gt": gt_p, "fit": fit_p,
            "rho_true": rho_true, "rho_obs": rho_obs, "rho_fit": rho_fit,
            "param_errors": param_errors, "psnr": psnr, "cc": cc,
        })

    # Average metrics
    avg_psnr = float(np.mean(all_psnr))
    avg_cc = float(np.mean(all_cc))
    print(f"\n[Summary] Avg PSNR = {avg_psnr:.2f} dB, Avg CC = {avg_cc:.4f}")

    metrics = {
        "PSNR": avg_psnr,
        "CC": avg_cc,
        "SSIM": "N/A (1D spectra)",
    }

    # Build gt_output and recon_output arrays
    gt_spectra = np.array([np.abs(r["rho_true"]) for r in all_results])
    recon_spectra = np.array([np.abs(r["rho_fit"]) for r in all_results])

    # Save
    print("[4] Saving outputs ...")
    for d in [RESULTS_DIR, ASSETS_DIR]:
        np.save(os.path.join(d, "gt_output.npy"), gt_spectra)
        np.save(os.path.join(d, "recon_output.npy"), recon_spectra)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    # Plot
    print("[5] Plotting ...")
    plot_results(freq, all_results, metrics)

    print(f"\n{'=' * 60}")
    print("Task 104 COMPLETE")
    print(f"{'=' * 60}")
    return metrics


if __name__ == "__main__":
    metrics = main()
