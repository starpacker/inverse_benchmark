"""
Nonlinear Photonic Inverse Design: 1D Transfer Matrix Method
=============================================================
Inverse problem: Given a target optical transmission/reflection spectrum,
optimize the dielectric permittivity distribution eps(x) of a 1D photonic
structure to match the target response.

Approach:
- Forward model: 1D Transfer Matrix Method (TMM) - exact for layered media
- Ground truth: Quarter-wave Bragg grating (analytically known)
- Optimization: L-BFGS-B with numerical gradients
- Target: Wavelength-selective filter (reflect at 1310nm, transmit at 1550nm)

Physics:
  For a stack of N layers with refractive indices n_j and thicknesses d_j,
  the transfer matrix for layer j at normal incidence is:
    M_j = [[cos(phi_j), -i sin(phi_j)/n_j], [-i n_j sin(phi_j), cos(phi_j)]]
  where phi_j = 2*pi*n_j*d_j / lambda is the phase accumulated in layer j.

  Total transfer matrix M = M_1 * M_2 * ... * M_N
  Transmission and reflection are computed from M and the boundary conditions.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import os
import json
import time

# ==============================================================================
# Physical constants and simulation parameters
# ==============================================================================
C0 = 3e8           # Speed of light [m/s]

# Design region parameters
N_DESIGN = 100      # Number of design layers
LAYER_THICKNESS = 30e-9  # Each layer thickness [m] (30 nm)

# Permittivity bounds
EPS_MIN = 1.0        # Air
EPS_MAX = 3.5        # Moderate-index material (n up to ~1.87)

# Refractive index of surrounding medium
N_IN = 1.0           # Input medium (air)
N_OUT = 1.0          # Output medium (air)

# Wavelengths of interest
LAMBDA_PASS = 1550e-9   # Passband wavelength [m]
LAMBDA_STOP = 1310e-9   # Stopband wavelength [m]

# Spectrum evaluation
N_WAVELENGTHS = 80
LAMBDA_MIN = 1100e-9
LAMBDA_MAX = 1800e-9

# Optimization
MAX_ITER = 400
N_FREQ_OPT = 60      # Number of frequencies used in optimization
N_MULTI_START = 5    # Number of random restarts


# ==============================================================================
# Transfer Matrix Method (TMM) Forward Solver
# ==============================================================================
def tmm_spectrum(eps_layers, wavelength):
    """
    Compute transmission and reflection using Transfer Matrix Method.

    Parameters:
        eps_layers: array of relative permittivities for each layer
        wavelength: wavelength [m]

    Returns:
        T: power transmission coefficient
        R: power reflection coefficient
    """
    n_layers = np.sqrt(np.maximum(eps_layers, 1e-6))  # Refractive indices
    k0 = 2 * np.pi / wavelength

    # Initialize total transfer matrix as identity
    M = np.eye(2, dtype=complex)

    for j in range(len(n_layers)):
        nj = n_layers[j]
        phi = k0 * nj * LAYER_THICKNESS

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        Mj = np.array([
            [cos_phi, -1j * sin_phi / nj],
            [-1j * nj * sin_phi, cos_phi]
        ], dtype=complex)

        M = M @ Mj

    # Transmission and reflection coefficients
    n_in = N_IN
    n_out = N_OUT

    denom = (M[0, 0] * n_in + M[0, 1] * n_in * n_out +
             M[1, 0] + M[1, 1] * n_out)
    t = 2 * n_in / denom
    r = (M[0, 0] * n_in + M[0, 1] * n_in * n_out -
         M[1, 0] - M[1, 1] * n_out) / denom

    T = float(np.real(np.abs(t) ** 2 * n_out / n_in))
    R = float(np.real(np.abs(r) ** 2))

    return T, R


def compute_field_distribution(eps_layers, wavelength, n_points_per_layer=5):
    """
    Compute electric field distribution through the multilayer stack.
    """
    n_layers = np.sqrt(np.maximum(eps_layers, 1e-6))
    k0 = 2 * np.pi / wavelength
    N = len(eps_layers)

    # Compute overall transfer matrix and transmission coefficient
    M_total = np.eye(2, dtype=complex)
    for j in range(N):
        nj = n_layers[j]
        phi = k0 * nj * LAYER_THICKNESS
        Mj = np.array([
            [np.cos(phi), -1j * np.sin(phi) / nj],
            [-1j * nj * np.sin(phi), np.cos(phi)]
        ], dtype=complex)
        M_total = M_total @ Mj

    denom = (M_total[0, 0] * N_IN + M_total[0, 1] * N_IN * N_OUT +
             M_total[1, 0] + M_total[1, 1] * N_OUT)
    t = 2 * N_IN / denom

    # At the output, only transmitted wave: E = t, H = n_out * t
    state_right = np.array([t, N_OUT * t], dtype=complex)

    # Propagate backward to find field at each interface
    interface_states = [None] * (N + 1)
    interface_states[N] = state_right

    for j in range(N - 1, -1, -1):
        nj = n_layers[j]
        phi = k0 * nj * LAYER_THICKNESS
        # Inverse transfer matrix
        M_inv = np.array([
            [np.cos(phi), 1j * np.sin(phi) / nj],
            [1j * nj * np.sin(phi), np.cos(phi)]
        ], dtype=complex)
        interface_states[j] = M_inv @ interface_states[j + 1]

    # Sample field within each layer
    x_all = []
    E_all = []

    for j in range(N):
        nj = n_layers[j]
        state_left = interface_states[j]

        for k in range(n_points_per_layer):
            frac = k / n_points_per_layer
            x = j * LAYER_THICKNESS + frac * LAYER_THICKNESS
            phi_k = k0 * nj * frac * LAYER_THICKNESS

            Mk = np.array([
                [np.cos(phi_k), -1j * np.sin(phi_k) / nj],
                [-1j * nj * np.sin(phi_k), np.cos(phi_k)]
            ], dtype=complex)

            field_here = Mk @ state_left
            x_all.append(x)
            E_all.append(field_here[0])

    x_all = np.array(x_all)
    E_all = np.array(E_all)

    sort_idx = np.argsort(x_all)
    return x_all[sort_idx], E_all[sort_idx]


def compute_spectrum(eps_design, wavelengths):
    """Compute transmission spectrum over multiple wavelengths."""
    T_spectrum = np.zeros(len(wavelengths))
    R_spectrum = np.zeros(len(wavelengths))
    for i, wl in enumerate(wavelengths):
        T, R = tmm_spectrum(eps_design, wl)
        T_spectrum[i] = T
        R_spectrum[i] = R
    return T_spectrum, R_spectrum


# ==============================================================================
# Ground Truth: Quarter-Wave Bragg Grating
# ==============================================================================
def create_bragg_grating(n_periods=8, n_high=2.2, n_low=1.45,
                         lambda_center=1310e-9):
    """
    Create a quarter-wave stack (Bragg grating) as ground truth.
    Each layer has optical thickness = lambda_center / 4.
    """
    eps_high = n_high ** 2
    eps_low = n_low ** 2

    # Physical thickness for quarter-wave condition
    t_high = lambda_center / (4 * n_high)
    t_low = lambda_center / (4 * n_low)

    # Convert to grid pixels
    pixels_high = max(1, int(round(t_high / LAYER_THICKNESS)))
    pixels_low = max(1, int(round(t_low / LAYER_THICKNESS)))

    eps_design = np.ones(N_DESIGN) * eps_low
    pos = 0

    for p in range(n_periods):
        for j in range(pixels_high):
            if pos < N_DESIGN:
                eps_design[pos] = eps_high
                pos += 1
        for j in range(pixels_low):
            if pos < N_DESIGN:
                eps_design[pos] = eps_low
                pos += 1

    return eps_design


# ==============================================================================
# Optimization
# ==============================================================================
def objective_with_gradient(eps_flat, wavelengths_opt, T_target_opt,
                            convergence_list, weights=None):
    """
    Compute objective and gradient via central finite differences.
    Objective: sum_i w_i * (T(eps, lambda_i) - T_target(lambda_i))^2
    """
    eps_design = np.clip(eps_flat, EPS_MIN, EPS_MAX)

    if weights is None:
        weights = np.ones(len(wavelengths_opt))

    # Forward evaluation
    T_current = np.zeros(len(wavelengths_opt))
    for i, wl in enumerate(wavelengths_opt):
        T, _ = tmm_spectrum(eps_design, wl)
        T_current[i] = T

    residuals = T_current - T_target_opt
    obj = float(np.sum(weights * residuals ** 2))
    convergence_list.append(obj)

    # Gradient via central finite differences (more accurate)
    delta = 1e-4
    grad = np.zeros(N_DESIGN)

    for k in range(N_DESIGN):
        eps_fwd = eps_design.copy()
        eps_bwd = eps_design.copy()
        eps_fwd[k] = min(eps_design[k] + delta, EPS_MAX + delta)
        eps_bwd[k] = max(eps_design[k] - delta, EPS_MIN - delta)

        T_fwd = np.zeros(len(wavelengths_opt))
        T_bwd = np.zeros(len(wavelengths_opt))
        for i, wl in enumerate(wavelengths_opt):
            T_fwd[i], _ = tmm_spectrum(eps_fwd, wl)
            T_bwd[i], _ = tmm_spectrum(eps_bwd, wl)

        obj_fwd = float(np.sum(weights * (T_fwd - T_target_opt) ** 2))
        obj_bwd = float(np.sum(weights * (T_bwd - T_target_opt) ** 2))
        grad[k] = (obj_fwd - obj_bwd) / (2 * delta)

    if len(convergence_list) % 10 == 1:
        print(f"  Eval {len(convergence_list):3d}: objective = {obj:.6f}")

    return obj, grad


def generate_initial_guesses(n_starts, wavelengths_opt, T_target_opt):
    """
    Generate diverse initial guesses for multi-start optimization.
    Includes physically-motivated initializations.
    """
    inits = []
    rng = np.random.RandomState(42)

    # 1. Uniform midpoint
    inits.append(np.ones(N_DESIGN) * (EPS_MIN + EPS_MAX) / 2.0)

    # 2-3. Random Bragg-like gratings with different parameters
    for n_high, n_low, n_per in [(1.7, 1.45, 7), (1.8, 1.4, 8), (1.6, 1.5, 6)]:
        if len(inits) >= n_starts:
            break
        eps_h = n_high ** 2
        eps_l = n_low ** 2
        t_h = LAMBDA_STOP / (4 * n_high)
        t_l = LAMBDA_STOP / (4 * n_low)
        px_h = max(1, int(round(t_h / LAYER_THICKNESS)))
        px_l = max(1, int(round(t_l / LAYER_THICKNESS)))
        eps = np.ones(N_DESIGN) * eps_l
        pos = 0
        for _ in range(n_per + 2):
            for _ in range(px_h):
                if pos < N_DESIGN:
                    eps[pos] = eps_h
                    pos += 1
            for _ in range(px_l):
                if pos < N_DESIGN:
                    eps[pos] = eps_l
                    pos += 1
        inits.append(eps)

    # 4+. Random smooth profiles
    while len(inits) < n_starts:
        base = rng.uniform(EPS_MIN + 0.3, EPS_MAX - 0.3, N_DESIGN)
        # Smooth with a moving average
        kernel = np.ones(5) / 5.0
        base = np.convolve(base, kernel, mode='same')
        base = np.clip(base, EPS_MIN, EPS_MAX)
        inits.append(base)

    return inits


def run_inverse_design(wavelengths_opt, T_target_opt, max_iter=MAX_ITER):
    """Run multi-start L-BFGS-B optimization for inverse design."""

    bounds = [(EPS_MIN, EPS_MAX)] * N_DESIGN

    # Compute spectral weights: emphasize stopband and passband regions
    weights = np.ones(len(wavelengths_opt))
    for i, wl in enumerate(wavelengths_opt):
        wl_nm = wl * 1e9
        # Extra weight near the stopband and passband centers
        if 1260 < wl_nm < 1360:
            weights[i] = 3.0
        elif 1500 < wl_nm < 1600:
            weights[i] = 2.0

    inits = generate_initial_guesses(N_MULTI_START, wavelengths_opt, T_target_opt)

    best_eps = None
    best_obj = np.inf
    best_convergence = []

    t_start = time.time()

    for s, eps_init in enumerate(inits):
        print(f"\n  --- Start {s+1}/{len(inits)} ---")
        convergence = []

        result = minimize(
            objective_with_gradient,
            eps_init,
            args=(wavelengths_opt, T_target_opt, convergence, weights),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'ftol': 1e-15,
                'gtol': 1e-12,
                'disp': False,
                'maxfun': max_iter * 4,
            }
        )

        print(f"  Start {s+1}: obj={result.fun:.6f}, evals={len(convergence)}, success={result.success}")

        if result.fun < best_obj:
            best_obj = result.fun
            best_eps = np.clip(result.x, EPS_MIN, EPS_MAX)
            best_convergence = convergence
            print(f"  *** New best! obj={best_obj:.6f} ***")

    t_elapsed = time.time() - t_start
    print(f"\nAll starts done in {t_elapsed:.1f}s")
    print(f"Best objective: {best_obj:.6f}")

    return best_eps, best_convergence


# ==============================================================================
# Evaluation Metrics
# ==============================================================================
def compute_psnr(gt, recon, data_range=None):
    """Peak Signal-to-Noise Ratio."""
    if data_range is None:
        data_range = np.max(gt) - np.min(gt)
    if data_range < 1e-10:
        data_range = 1.0
    mse = np.mean((gt - recon) ** 2)
    if mse < 1e-30:
        return 100.0
    return 10 * np.log10(data_range ** 2 / mse)


def compute_correlation(gt, recon):
    """Pearson correlation coefficient."""
    gt_c = gt - np.mean(gt)
    recon_c = recon - np.mean(recon)
    num = np.sum(gt_c * recon_c)
    den = np.sqrt(np.sum(gt_c ** 2) * np.sum(recon_c ** 2))
    if den < 1e-30:
        return 0.0
    return float(num / den)


def compute_spectral_rmse(T1, T2):
    """Root mean squared error between two spectra."""
    return float(np.sqrt(np.mean((T1 - T2) ** 2)))


# ==============================================================================
# Main
# ==============================================================================
def main():
    print("=" * 70)
    print("1D Photonic Inverse Design: Bragg Grating Filter (TMM)")
    print("=" * 70)

    os.makedirs('results', exist_ok=True)

    # ---- Step 1: Ground truth ----
    print("\n[1/5] Creating ground truth Bragg grating...")
    eps_gt = create_bragg_grating(
        n_periods=7, n_high=1.70, n_low=1.45,
        lambda_center=LAMBDA_STOP
    )
    print(f"  GT: {N_DESIGN} layers, eps in [{eps_gt.min():.2f}, {eps_gt.max():.2f}]")

    # ---- Step 2: Target spectrum ----
    print("\n[2/5] Computing target spectrum...")
    wavelengths = np.linspace(LAMBDA_MIN, LAMBDA_MAX, N_WAVELENGTHS)
    T_target, R_target = compute_spectrum(eps_gt, wavelengths)

    T_pass = float(np.interp(LAMBDA_PASS, wavelengths, T_target))
    T_stop = float(np.interp(LAMBDA_STOP, wavelengths, T_target))
    print(f"  Target T@{LAMBDA_PASS*1e9:.0f}nm = {T_pass:.4f}")
    print(f"  Target T@{LAMBDA_STOP*1e9:.0f}nm = {T_stop:.4f}")

    # ---- Step 3: Optimization wavelengths ----
    print("\n[3/5] Setting up optimization...")
    # Dense sampling: passband, stopband, and broadband
    n_pass = N_FREQ_OPT // 4
    n_stop = N_FREQ_OPT // 4
    n_broad = N_FREQ_OPT - n_pass - n_stop
    wl_pass = np.linspace(1450e-9, 1650e-9, n_pass)
    wl_stop = np.linspace(1220e-9, 1400e-9, n_stop)
    wl_broad = np.linspace(1100e-9, 1800e-9, n_broad)
    wavelengths_opt = np.sort(np.unique(np.concatenate([wl_pass, wl_stop, wl_broad])))

    T_target_opt = np.zeros(len(wavelengths_opt))
    for i, wl in enumerate(wavelengths_opt):
        T_target_opt[i] = float(np.interp(wl, wavelengths, T_target))

    print(f"  {len(wavelengths_opt)} optimization wavelengths")

    # ---- Step 4: Inverse design ----
    print("\n[4/5] Running inverse design...")
    eps_optimized, convergence = run_inverse_design(
        wavelengths_opt, T_target_opt, MAX_ITER
    )
    print(f"  Optimized eps in [{eps_optimized.min():.2f}, {eps_optimized.max():.2f}]")

    # ---- Step 5: Evaluate ----
    print("\n[5/5] Evaluating results...")
    T_achieved, R_achieved = compute_spectrum(eps_optimized, wavelengths)

    T_opt_pass = float(np.interp(LAMBDA_PASS, wavelengths, T_achieved))
    T_opt_stop = float(np.interp(LAMBDA_STOP, wavelengths, T_achieved))
    print(f"  Achieved T@{LAMBDA_PASS*1e9:.0f}nm = {T_opt_pass:.4f}")
    print(f"  Achieved T@{LAMBDA_STOP*1e9:.0f}nm = {T_opt_stop:.4f}")

    structure_psnr = compute_psnr(eps_gt, eps_optimized, data_range=EPS_MAX - EPS_MIN)
    structure_cc = compute_correlation(eps_gt, eps_optimized)
    spectral_rmse = compute_spectral_rmse(T_target, T_achieved)
    spectrum_cc = compute_correlation(T_target, T_achieved)

    print(f"  Structure PSNR: {structure_psnr:.2f} dB")
    print(f"  Structure CC:   {structure_cc:.4f}")
    print(f"  Spectral RMSE:  {spectral_rmse:.6f}")
    print(f"  Spectrum CC:    {spectrum_cc:.4f}")

    # Field distribution
    x_field, E_field = compute_field_distribution(eps_optimized, LAMBDA_PASS)

    # ---- Save outputs ----
    np.save('results/ground_truth.npy', eps_gt)
    np.save('results/reconstruction.npy', eps_optimized)
    # Also save as gt_output.npy / recon_output.npy at top level for website
    np.save('gt_output.npy', eps_gt)
    np.save('recon_output.npy', eps_optimized)

    metrics = {
        'task': 'angler_photonic',
        'method': '1D_TMM_inverse_design',
        'structure_psnr_dB': round(float(structure_psnr), 4),
        'structure_correlation': round(float(structure_cc), 4),
        'spectral_rmse': round(float(spectral_rmse), 6),
        'spectrum_correlation': round(float(spectrum_cc), 4),
        'transmission_at_1550nm_target': round(float(T_pass), 4),
        'transmission_at_1550nm_achieved': round(float(T_opt_pass), 4),
        'transmission_at_1310nm_target': round(float(T_stop), 4),
        'transmission_at_1310nm_achieved': round(float(T_opt_stop), 4),
        'n_design_layers': N_DESIGN,
        'n_optimization_iterations': len(convergence),
        'final_objective': round(float(convergence[-1]), 6) if convergence else None,
        'initial_objective': round(float(convergence[0]), 6) if convergence else None,
    }

    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to results/metrics.json")

    # ---- Visualization ----
    print("  Generating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Permittivity profiles
    ax = axes[0, 0]
    x_design = np.arange(N_DESIGN) * LAYER_THICKNESS * 1e6
    ax.step(x_design, eps_gt, 'b-', label='Ground Truth', linewidth=1.5,
            where='mid')
    ax.step(x_design, eps_optimized, 'r--', label='Optimized', linewidth=1.5,
            alpha=0.8, where='mid')
    ax.set_xlabel('Position (um)', fontsize=12)
    ax.set_ylabel('Relative Permittivity', fontsize=12)
    ax.set_title(
        f'(a) Permittivity Profile\n(PSNR={structure_psnr:.1f} dB, CC={structure_cc:.3f})',
        fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim([0, EPS_MAX + 0.5])
    ax.grid(True, alpha=0.3)

    # (b) Transmission spectrum
    ax = axes[0, 1]
    wl_nm = wavelengths * 1e9
    ax.plot(wl_nm, T_target, 'b-', label='Target (GT)', linewidth=2)
    ax.plot(wl_nm, T_achieved, 'r--', label='Achieved', linewidth=2, alpha=0.8)
    ax.axvline(x=1550, color='green', linestyle=':', alpha=0.6,
               label='pass=1550nm')
    ax.axvline(x=1310, color='orange', linestyle=':', alpha=0.6,
               label='stop=1310nm')
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Transmission', fontsize=12)
    ax.set_title(
        f'(b) Transmission Spectrum\n(RMSE={spectral_rmse:.4f}, CC={spectrum_cc:.3f})',
        fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.set_ylim([-0.05, 1.15])
    ax.grid(True, alpha=0.3)

    # (c) Convergence
    ax = axes[1, 0]
    if len(convergence) > 1:
        ax.semilogy(range(1, len(convergence) + 1), convergence, 'k-',
                    linewidth=1.5)
    else:
        ax.plot([1], convergence if convergence else [0], 'ko')
    ax.set_xlabel('Function Evaluation', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title('(c) Optimization Convergence', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    # (d) Electric field distribution
    ax = axes[1, 1]
    if len(x_field) > 0:
        x_um = x_field * 1e6
        E_intensity = np.abs(E_field) ** 2
        E_norm = E_intensity / (np.max(E_intensity) + 1e-30)
        ax.plot(x_um, E_norm, 'b-', linewidth=1.0, alpha=0.8)
        ax.fill_between(x_um, 0, E_norm, alpha=0.15, color='blue')

        # Overlay permittivity profile (scaled)
        x_eps = np.arange(N_DESIGN) * LAYER_THICKNESS * 1e6
        eps_scaled = eps_optimized / EPS_MAX
        ax.step(x_eps, eps_scaled, 'gray', linewidth=0.8, alpha=0.5,
                where='mid', label='eps (scaled)')

    ax.set_xlabel('Position (um)', fontsize=12)
    ax.set_ylabel('|E|^2 (normalized)', fontsize=12)
    ax.set_title(f'(d) Field at lambda={LAMBDA_PASS*1e9:.0f}nm',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Photonic Inverse Design: 1D Bragg Grating (TMM)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/reconstruction_result.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved results/reconstruction_result.png")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Structure PSNR:        {structure_psnr:.2f} dB")
    print(f"  Structure CC:          {structure_cc:.4f}")
    print(f"  Spectral RMSE:         {spectral_rmse:.6f}")
    print(f"  Spectrum CC:           {spectrum_cc:.4f}")
    print(f"  T@1550nm (target):     {T_pass:.4f}")
    print(f"  T@1550nm (achieved):   {T_opt_pass:.4f}")
    print(f"  T@1310nm (target):     {T_stop:.4f}")
    print(f"  T@1310nm (achieved):   {T_opt_stop:.4f}")
    print(f"  Iterations:            {len(convergence)}")
    if convergence:
        print(f"  Final objective:       {convergence[-1]:.6f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
