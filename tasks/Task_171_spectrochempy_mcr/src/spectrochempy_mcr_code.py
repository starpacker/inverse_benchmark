#!/usr/bin/env python
"""
MCR-ALS Spectral Decomposition: Inverse Problem
=================================================
Decompose a mixed spectral matrix D into pure component spectra S
and concentration profiles C, such that D = C @ S.T

Forward model:  D = C @ S^T + noise
Inverse solver: MCR-ALS (Multivariate Curve Resolution - Alternating Least Squares)
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from scipy.signal import find_peaks
import json
import os

# ─── reproducibility ───
np.random.seed(42)

# ─── output directory ───
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 1. Synthesize pure component spectra and concentration data
# ============================================================

def gaussian_peak(wavelengths, center, width, amplitude=1.0):
    """Generate a Gaussian peak."""
    return amplitude * np.exp(-0.5 * ((wavelengths - center) / width) ** 2)


def create_pure_spectra(wavelengths):
    """Create 3 pure component spectra with Gaussian peaks."""
    n_wl = len(wavelengths)
    S = np.zeros((3, n_wl))

    # S1: peaks at 250 nm and 450 nm
    S[0] = gaussian_peak(wavelengths, 250, 20, 1.0) + \
           gaussian_peak(wavelengths, 450, 25, 0.8)

    # S2: peaks at 350 nm and 550 nm
    S[1] = gaussian_peak(wavelengths, 350, 22, 0.9) + \
           gaussian_peak(wavelengths, 550, 28, 1.0)

    # S3: peaks at 300 nm, 500 nm, and 650 nm
    S[2] = gaussian_peak(wavelengths, 300, 18, 0.7) + \
           gaussian_peak(wavelengths, 500, 20, 0.85) + \
           gaussian_peak(wavelengths, 650, 22, 0.6)

    return S


def create_concentration_profiles(n_samples, n_components):
    """Create sinusoidal concentration profiles (simulating kinetics)."""
    t = np.linspace(0, 2 * np.pi, n_samples)
    C = np.zeros((n_samples, n_components))

    # Component 1: decaying sinusoidal
    C[:, 0] = 0.5 * (1 + np.sin(t)) * np.exp(-0.2 * t) + 0.1

    # Component 2: growing then decaying
    C[:, 1] = 0.8 * np.sin(t + np.pi / 3) ** 2 + 0.05

    # Component 3: delayed growth
    C[:, 2] = 0.6 * (1 - np.exp(-0.5 * t)) * np.abs(np.cos(t / 2)) + 0.1

    # Ensure non-negativity
    C = np.maximum(C, 0)
    return C


# ============================================================
# 2. Forward model: D = C @ S^T + noise
# ============================================================

def forward_model(C, S, snr_db=30):
    """
    Forward model: generate mixed spectral data with additive Gaussian noise.
    D = C @ S^T + noise
    """
    D_clean = C @ S
    signal_power = np.mean(D_clean ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), D_clean.shape)
    D_noisy = D_clean + noise
    return D_clean, D_noisy


# ============================================================
# 3. Inverse solver: MCR-ALS
# ============================================================

def simplisma_init(D, n_components):
    """
    SIMPLISMA-like initialization: pick the purest rows/columns
    to initialize component spectra.
    """
    # Use SVD for a more robust initialization
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    S_init = np.abs(Vt[:n_components, :])
    # Normalize each spectrum
    for i in range(n_components):
        S_init[i] /= (np.max(S_init[i]) + 1e-12)
    return S_init


def mcr_als_manual(D, n_components, max_iter=500, tol=1e-6):
    """
    Manual MCR-ALS implementation with non-negativity constraints.

    Algorithm:
        1. Initialize S via SVD
        2. Repeat until convergence:
            a. Fix S, solve for C using NNLS (column-wise)
            b. Fix C, solve for S using NNLS (column-wise)
            c. Check convergence (relative change in lack-of-fit)
    """
    n_samples, n_wavelengths = D.shape

    # Initialize spectra via SVD
    S = simplisma_init(D, n_components)

    lof_prev = np.inf
    lof_history = []

    for iteration in range(max_iter):
        # Step A: Fix S, solve for C using NNLS
        # D = C @ S  =>  D^T = S^T @ C^T
        # For each sample i: D[i,:] = C[i,:] @ S  => solve NNLS
        C = np.zeros((n_samples, n_components))
        for i in range(n_samples):
            C[i, :], _ = nnls(S.T, D[i, :])

        # Step B: Fix C, solve for S using NNLS
        # D = C @ S  =>  D^T = S^T @ C^T
        # For each wavelength j: D[:,j] = C @ S[:,j] => solve NNLS
        S_new = np.zeros((n_components, n_wavelengths))
        for j in range(n_wavelengths):
            S_new[:, j], _ = nnls(C, D[:, j])
        S = S_new

        # Compute lack-of-fit
        D_reconstructed = C @ S
        residual = D - D_reconstructed
        lof = np.sqrt(np.sum(residual ** 2) / np.sum(D ** 2)) * 100  # percentage

        lof_history.append(lof)

        # Check convergence
        if abs(lof_prev - lof) < tol:
            print(f"MCR-ALS converged at iteration {iteration + 1}, LOF = {lof:.4f}%")
            break
        lof_prev = lof

    else:
        print(f"MCR-ALS reached max iterations ({max_iter}), LOF = {lof:.4f}%")

    return C, S, lof_history


def try_pymcr(D, n_components):
    """Try using pyMCR for MCR-ALS decomposition."""
    try:
        from pymcr.mcr import McrAR
        from pymcr.constraints import ConstraintNonneg, ConstraintNorm
        from pymcr.regressors import OLS, NNLS

        # SVD-based initial guess
        U, s, Vt = np.linalg.svd(D, full_matrices=False)
        S_init = np.abs(Vt[:n_components, :])

        mcr = McrAR(
            max_iter=500,
            tol_increase=50,
            tol_n_increase=20,
            tol_err_change=1e-8,
            c_regr=NNLS(),
            st_regr=NNLS(),
            c_constraints=[ConstraintNonneg()],
            st_constraints=[ConstraintNonneg(), ConstraintNorm()]
        )
        mcr.fit(D, ST=S_init)
        C_recovered = mcr.C_opt_
        S_recovered = mcr.ST_opt_
        print(f"pyMCR converged. Final error: {mcr.err_[-1]:.6f}")
        return C_recovered, S_recovered, True
    except Exception as e:
        print(f"pyMCR failed: {e}")
        return None, None, False


# ============================================================
# 4. Evaluation metrics
# ============================================================

def match_components(S_true, S_recovered):
    """
    Match recovered components to true components using correlation.
    Returns permutation indices and sign flips.
    """
    n_comp = S_true.shape[0]
    corr_matrix = np.zeros((n_comp, n_comp))

    for i in range(n_comp):
        for j in range(n_comp):
            s_true_norm = S_true[i] / (np.linalg.norm(S_true[i]) + 1e-12)
            s_rec_norm = S_recovered[j] / (np.linalg.norm(S_recovered[j]) + 1e-12)
            corr_matrix[i, j] = np.abs(np.dot(s_true_norm, s_rec_norm))

    # Greedy matching
    perm = []
    used = set()
    for i in range(n_comp):
        best_j = -1
        best_corr = -1
        for j in range(n_comp):
            if j not in used and corr_matrix[i, j] > best_corr:
                best_corr = corr_matrix[i, j]
                best_j = j
        perm.append(best_j)
        used.add(best_j)

    return perm, corr_matrix


def compute_spectral_cc(S_true, S_recovered, perm):
    """Compute correlation coefficients between true and recovered spectra."""
    ccs = []
    for i, j in enumerate(perm):
        cc = np.corrcoef(S_true[i], S_recovered[j])[0, 1]
        ccs.append(abs(cc))
    return np.array(ccs)


def compute_concentration_re(C_true, C_recovered, perm):
    """Compute relative error of recovered concentrations."""
    res = []
    for i, j in enumerate(perm):
        # Scale recovered to match true (MCR has scale ambiguity)
        c_true = C_true[:, i]
        c_rec = C_recovered[:, j]
        # Find optimal scaling factor
        scale = np.dot(c_true, c_rec) / (np.dot(c_rec, c_rec) + 1e-12)
        c_rec_scaled = c_rec * scale
        re = np.linalg.norm(c_true - c_rec_scaled) / (np.linalg.norm(c_true) + 1e-12)
        res.append(re)
    return np.array(res)


def compute_psnr(D_true, D_reconstructed):
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((D_true - D_reconstructed) ** 2)
    if mse < 1e-15:
        return 100.0
    max_val = np.max(np.abs(D_true))
    psnr = 10 * np.log10(max_val ** 2 / mse)
    return psnr


def compute_lack_of_fit(D, D_reconstructed):
    """Compute lack-of-fit percentage."""
    residual = D - D_reconstructed
    lof = np.sqrt(np.sum(residual ** 2) / np.sum(D ** 2)) * 100
    return lof


# ============================================================
# 5. Visualization
# ============================================================

def plot_results(wavelengths, S_true, S_recovered, C_true, C_recovered,
                 D_clean, D_reconstructed, perm, save_path):
    """Create 4-subplot visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['#e74c3c', '#2ecc71', '#3498db']
    comp_names = ['Component 1', 'Component 2', 'Component 3']

    # Scale recovered components to match true for visualization
    S_rec_scaled = np.zeros_like(S_recovered)
    C_rec_scaled = np.zeros_like(C_recovered)
    for i, j in enumerate(perm):
        s_scale = np.max(S_true[i]) / (np.max(S_recovered[j]) + 1e-12)
        S_rec_scaled[i] = S_recovered[j] * s_scale

        c_scale = np.dot(C_true[:, i], C_recovered[:, j]) / (np.dot(C_recovered[:, j], C_recovered[:, j]) + 1e-12)
        C_rec_scaled[:, i] = C_recovered[:, j] * c_scale

    # (a) True vs recovered spectra
    ax = axes[0, 0]
    for i in range(S_true.shape[0]):
        ax.plot(wavelengths, S_true[i], color=colors[i], linewidth=2,
                label=f'{comp_names[i]} (true)')
        ax.plot(wavelengths, S_rec_scaled[i], color=colors[i], linewidth=1.5,
                linestyle='--', label=f'{comp_names[i]} (recovered)')
    ax.set_xlabel('Wavelength (nm)', fontsize=11)
    ax.set_ylabel('Intensity', fontsize=11)
    ax.set_title('(a) True vs Recovered Component Spectra', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

    # (b) True vs recovered concentration profiles
    ax = axes[0, 1]
    samples = np.arange(C_true.shape[0])
    for i in range(C_true.shape[1]):
        ax.plot(samples, C_true[:, i], color=colors[i], linewidth=2,
                label=f'{comp_names[i]} (true)')
        ax.plot(samples, C_rec_scaled[:, i], color=colors[i], linewidth=1.5,
                linestyle='--', label=f'{comp_names[i]} (recovered)')
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Concentration', fontsize=11)
    ax.set_title('(b) True vs Recovered Concentration Profiles', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

    # (c) Reconstructed vs original D (first 5 samples)
    ax = axes[1, 0]
    n_show = min(5, D_clean.shape[0])
    sample_colors = plt.cm.viridis(np.linspace(0, 1, n_show))
    for i in range(n_show):
        ax.plot(wavelengths, D_clean[i], color=sample_colors[i], linewidth=1.5,
                label=f'Sample {i+1} (true)')
        ax.plot(wavelengths, D_reconstructed[i], color=sample_colors[i],
                linewidth=1, linestyle='--', alpha=0.8)
    ax.set_xlabel('Wavelength (nm)', fontsize=11)
    ax.set_ylabel('Intensity', fontsize=11)
    ax.set_title('(c) Original vs Reconstructed Spectra', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # (d) Residual matrix heatmap
    ax = axes[1, 1]
    residual = D_clean - D_reconstructed
    im = ax.imshow(residual, aspect='auto', cmap='RdBu_r',
                   extent=[wavelengths[0], wavelengths[-1], D_clean.shape[0], 0])
    ax.set_xlabel('Wavelength (nm)', fontsize=11)
    ax.set_ylabel('Sample Index', fontsize=11)
    ax.set_title('(d) Residual Matrix (D_true - D_reconstructed)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Residual')

    plt.suptitle('MCR-ALS Spectral Decomposition Results', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")


# ============================================================
# Main execution
# ============================================================

def main():
    # ── Parameters ──
    n_samples = 20
    n_components = 3
    n_wavelengths = 200
    snr_db = 30
    wavelengths = np.linspace(200, 700, n_wavelengths)

    # ── 1. Create ground truth ──
    print("=" * 60)
    print("MCR-ALS Spectral Decomposition")
    print("=" * 60)

    S_true = create_pure_spectra(wavelengths)          # (3, 200)
    C_true = create_concentration_profiles(n_samples, n_components)  # (20, 3)

    print(f"Pure spectra matrix S: {S_true.shape}")
    print(f"Concentration matrix C: {C_true.shape}")

    # ── 2. Forward model ──
    D_clean, D_noisy = forward_model(C_true, S_true, snr_db=snr_db)
    print(f"Data matrix D: {D_noisy.shape}")
    print(f"SNR: {snr_db} dB")

    actual_snr = 10 * np.log10(np.mean(D_clean**2) / np.mean((D_noisy - D_clean)**2))
    print(f"Actual SNR: {actual_snr:.2f} dB")

    # ── 3. Inverse: MCR-ALS ──
    print("\n--- Attempting pyMCR ---")
    C_pymcr, S_pymcr, pymcr_ok = try_pymcr(D_noisy, n_components)

    if pymcr_ok:
        C_recovered = C_pymcr
        S_recovered = S_pymcr
        method_used = "pyMCR"
    else:
        print("\n--- Falling back to manual MCR-ALS ---")
        C_recovered, S_recovered, lof_hist = mcr_als_manual(D_noisy, n_components,
                                                             max_iter=500, tol=1e-7)
        method_used = "Manual MCR-ALS"

    print(f"\nMethod used: {method_used}")
    print(f"Recovered C: {C_recovered.shape}, S: {S_recovered.shape}")

    # ── 4. Evaluate ──
    # Match components
    perm, corr_matrix = match_components(S_true, S_recovered)
    print(f"Component matching (true→recovered): {perm}")

    # Spectral correlation coefficients
    spectral_ccs = compute_spectral_cc(S_true, S_recovered, perm)
    mean_cc = np.mean(spectral_ccs)
    print(f"\nSpectral Correlation Coefficients:")
    for i, cc in enumerate(spectral_ccs):
        print(f"  Component {i+1}: {cc:.6f}")
    print(f"  Mean CC: {mean_cc:.6f}")

    # Concentration relative error
    conc_res = compute_concentration_re(C_true, C_recovered, perm)
    mean_re = np.mean(conc_res)
    print(f"\nConcentration Relative Errors:")
    for i, re in enumerate(conc_res):
        print(f"  Component {i+1}: {re:.6f}")
    print(f"  Mean RE: {mean_re:.6f}")

    # Reconstruction
    D_reconstructed = C_recovered @ S_recovered
    psnr = compute_psnr(D_clean, D_reconstructed)
    lof = compute_lack_of_fit(D_clean, D_reconstructed)
    print(f"\nReconstruction PSNR: {psnr:.2f} dB")
    print(f"Lack-of-fit: {lof:.4f}%")

    # ── 5. Save results ──
    metrics = {
        "task": "spectrochempy_mcr",
        "method": method_used,
        "inverse_problem": "MCR-ALS spectral decomposition",
        "n_samples": n_samples,
        "n_components": n_components,
        "n_wavelengths": n_wavelengths,
        "snr_db": snr_db,
        "spectral_cc_per_component": spectral_ccs.tolist(),
        "spectral_cc_mean": float(mean_cc),
        "concentration_re_per_component": conc_res.tolist(),
        "concentration_re_mean": float(mean_re),
        "reconstruction_psnr_db": float(psnr),
        "lack_of_fit_pct": float(lof),
        "component_matching": perm
    }

    with open(os.path.join(RESULTS_DIR, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {RESULTS_DIR}/metrics.json")

    # Save ground truth: D_clean (noise-free data matrix)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), D_clean)

    # Save reconstruction: D_reconstructed from MCR-ALS
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), D_reconstructed)

    print(f"Ground truth saved: {RESULTS_DIR}/ground_truth.npy {D_clean.shape}")
    print(f"Reconstruction saved: {RESULTS_DIR}/reconstruction.npy {D_reconstructed.shape}")

    # ── 6. Visualization ──
    plot_results(wavelengths, S_true, S_recovered, C_true, C_recovered,
                 D_clean, D_reconstructed, perm,
                 os.path.join(RESULTS_DIR, "reconstruction_result.png"))

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Mean Spectral CC: {mean_cc:.6f} (target > 0.95)")
    print(f"Reconstruction PSNR: {psnr:.2f} dB (target > 25)")
    print(f"Mean Concentration RE: {mean_re:.6f}")
    print(f"Lack-of-fit: {lof:.4f}%")

    if mean_cc > 0.95 and psnr > 25:
        print("✓ All targets met!")
    else:
        print("✗ Some targets not met.")


if __name__ == "__main__":
    main()
