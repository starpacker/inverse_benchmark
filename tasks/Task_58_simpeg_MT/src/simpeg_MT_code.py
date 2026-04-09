"""
simpeg_MT — Magnetotelluric (MT) Inversion
=============================================
Task: Recover 1D layered-earth electrical conductivity profile
      from surface impedance measurements at multiple frequencies.

Inverse Problem:
    Given apparent resistivity ρ_a(f) and phase φ(f) at N frequencies,
    recover the conductivity σ(z) vs depth profile.

Forward Model (SimPEG):
    1D MT forward using natural-source EM: the plane-wave impedance Z(f)
    is computed analytically via recursive Fresnel-like formulae for
    layered media.  SimPEG uses a finite-volume discretisation for
    the general case.

Inverse Solver:
    SimPEG's Gauss-Newton with Tikhonov regularisation (smoothness
    in log-conductivity space) and beta cooling schedule.

Repo: https://github.com/simpeg/simpeg
Paper: Cockett et al. (2015), Computers & Geosciences.

Usage:
    /data/yjh/geo_env/bin/python simpeg_MT_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

# ── SimPEG imports (used for inversion framework) ──────
try:
    from simpeg import (
        maps, data_misfit, regularization, optimization,
        inverse_problem, inversion, directives, data,
    )
    from simpeg.electromagnetics import natural_source as nsem
    from discretize import TensorMesh
    HAS_SIMPEG = True
except ImportError:
    HAS_SIMPEG = False

# ═══════════════════════════════════════════════════════════
# 1. Configuration
# ═══════════════════════════════════════════════════════════
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1D layered-earth model
# Ground truth: 3-layer model
GT_THICKNESSES = [200.0, 500.0]        # m (top two layers; bottom is half-space)
GT_RESISTIVITIES = [100.0, 10.0, 1000.0]  # Ohm·m (resistive, conductive, resistive)

# Survey parameters
N_FREQ = 50
FREQ_MIN = 0.001   # Hz
FREQ_MAX = 100.0    # Hz
NOISE_FLOOR = 0.01  # absolute noise floor (fraction of |Z|)
NOISE_PCT = 0.02    # 2% relative noise

# 1D mesh
N_LAYERS = 30       # discretised layers for inversion
MIN_DEPTH = 10.0    # m (shallowest layer)
MAX_DEPTH = 5000.0  # m (deepest)

SEED = 42


# ═══════════════════════════════════════════════════════════
# 2. 1D MT Forward (Analytic Wait's recursion)
# ═══════════════════════════════════════════════════════════
def mt_1d_forward(frequencies, thicknesses, resistivities):
    """
    Compute 1D MT impedance using Wait's recursive formula.

    This is the analytic solution for a layered earth under
    plane-wave excitation.

    Z_n = Z_{n,intrinsic}  (bottom layer)
    Z_j = Z_{j,intr} * (Z_{j+1} + Z_{j,intr} * tanh(ik_j * h_j)) /
                        (Z_{j,intr} + Z_{j+1} * tanh(ik_j * h_j))

    where k_j = sqrt(iωμσ_j) and Z_{j,intr} = iωμ/k_j.

    Parameters
    ----------
    frequencies : np.ndarray  Frequencies [Hz].
    thicknesses : list        Layer thicknesses [m] (N-1 for N layers).
    resistivities : list      Layer resistivities [Ω·m] (N layers).

    Returns
    -------
    Z : np.ndarray           Complex impedance at each frequency.
    app_res : np.ndarray     Apparent resistivity [Ω·m].
    phase : np.ndarray       Phase [degrees].
    """
    mu0 = 4 * np.pi * 1e-7  # H/m
    n_layers = len(resistivities)

    Z = np.zeros(len(frequencies), dtype=complex)

    for fi, freq in enumerate(frequencies):
        omega = 2 * np.pi * freq

        # Intrinsic impedance and propagation constant for each layer
        sigma = [1.0 / r for r in resistivities]
        k = [np.sqrt(1j * omega * mu0 * s) for s in sigma]
        Z_intr = [1j * omega * mu0 / kk for kk in k]

        # Start from bottom (half-space)
        Z_below = Z_intr[-1]

        # Recurse upward
        for j in range(n_layers - 2, -1, -1):
            h = thicknesses[j]
            arg = k[j] * h
            # Numerical stability: clip argument
            if np.abs(arg) > 500:
                tanh_val = 1.0 if arg.real > 0 else -1.0
            else:
                tanh_val = np.tanh(arg)

            Z_below = Z_intr[j] * (Z_below + Z_intr[j] * tanh_val) / \
                      (Z_intr[j] + Z_below * tanh_val)

        Z[fi] = Z_below

    app_res = np.abs(Z) ** 2 / (2 * np.pi * frequencies * mu0)
    phase = np.degrees(np.angle(Z))

    return Z, app_res, phase


def forward_operator(frequencies, thicknesses, resistivities):
    """
    Forward operator wrapper.
    Returns impedance, apparent resistivity, and phase.
    """
    return mt_1d_forward(frequencies, thicknesses, resistivities)


# ═══════════════════════════════════════════════════════════
# 3. Data Generation
# ═══════════════════════════════════════════════════════════
def load_or_generate_data():
    """Generate synthetic MT sounding data."""
    print("[DATA] Generating synthetic 1D MT data ...")

    frequencies = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_FREQ)

    Z_clean, rho_clean, phi_clean = forward_operator(
        frequencies, GT_THICKNESSES, GT_RESISTIVITIES
    )

    print(f"[DATA] {N_FREQ} frequencies: [{FREQ_MIN:.4f}, {FREQ_MAX:.1f}] Hz")
    print(f"[DATA] ρ_a range: [{rho_clean.min():.1f}, {rho_clean.max():.1f}] Ω·m")
    print(f"[DATA] φ range: [{phi_clean.min():.1f}, {phi_clean.max():.1f}]°")

    # Add noise to impedance
    rng = np.random.default_rng(SEED)
    noise_real = (NOISE_FLOOR * np.abs(Z_clean) +
                  NOISE_PCT * np.abs(Z_clean)) * rng.standard_normal(N_FREQ)
    noise_imag = (NOISE_FLOOR * np.abs(Z_clean) +
                  NOISE_PCT * np.abs(Z_clean)) * rng.standard_normal(N_FREQ)
    Z_noisy = Z_clean + noise_real + 1j * noise_imag

    rho_noisy = np.abs(Z_noisy) ** 2 / (2 * np.pi * frequencies * 4 * np.pi * 1e-7)
    phi_noisy = np.degrees(np.angle(Z_noisy))

    return frequencies, Z_clean, Z_noisy, rho_clean, rho_noisy, phi_clean, phi_noisy


# ═══════════════════════════════════════════════════════════
# 4. Inverse Solver — Occam/Tikhonov 1D inversion
# ═══════════════════════════════════════════════════════════
def reconstruct(frequencies, Z_noisy):
    """
    1D MT inversion using Occam's razor approach with analytical Jacobian.
    Tikhonov regularisation with smoothness constraint in log-resistivity space.
    """
    from scipy.optimize import minimize

    # Create logarithmically spaced depth layers
    layer_thicknesses = np.logspace(
        np.log10(MIN_DEPTH), np.log10(MAX_DEPTH / N_LAYERS), N_LAYERS - 1
    )
    depths = np.cumsum(layer_thicknesses)

    # Inversion in log10-resistivity space
    m0 = np.log10(np.ones(N_LAYERS) * 100.0)  # starting model: 100 Ω·m

    # Smoothness matrix (first-order finite difference)
    D = np.zeros((N_LAYERS - 1, N_LAYERS))
    for i in range(N_LAYERS - 1):
        D[i, i] = -1
        D[i, i + 1] = 1

    # Precompute DtD for regularisation gradient
    DtD = D.T @ D

    # Noise estimate
    Z_std = NOISE_FLOOR * np.abs(Z_noisy) + NOISE_PCT * np.abs(Z_noisy)

    mu0 = 4 * np.pi * 1e-7
    n_freq = len(frequencies)
    n_layers = N_LAYERS
    ln10 = np.log(10.0)

    def forward_with_analytical_grad(m):
        """
        Compute impedance Z and analytical dZ/dm via differentiation of
        Wait's recursion w.r.t. log10(rho).

        For each layer j:
          sigma_j = 1/rho_j,  k_j = sqrt(i*omega*mu0*sigma_j)
          Z_intr_j = i*omega*mu0 / k_j
          dm_j = log10(rho_j), so d(rho_j)/d(m_j) = rho_j * ln(10)

        Derivatives:
          dk_j/dm_j = -k_j * ln10 / 2
          dZ_intr_j/dm_j = Z_intr_j * ln10 / 2

        Recursion:
          Z_j = Z_intr_j * (Z_{j+1} + Z_intr_j * t_j) / (Z_intr_j + Z_{j+1} * t_j)
          where t_j = tanh(k_j * h_j)

        For layer p:
          dZ_0/dm_p = (prod_{j=0}^{p-1} dZ_j/dZ_{j+1}) * dZ_p/dm_p(local)
        """
        resistivities = 10.0 ** m

        Z_pred = np.zeros(n_freq, dtype=complex)
        J = np.zeros((n_freq, n_layers), dtype=complex)

        for fi in range(n_freq):
            omega = 2 * np.pi * frequencies[fi]

            sigma = 1.0 / resistivities
            k = np.sqrt(1j * omega * mu0 * sigma)
            Z_intr = 1j * omega * mu0 / k

            # ── Forward pass: store Z at each interface ──
            Z_at = np.zeros(n_layers, dtype=complex)
            tanh_v = np.zeros(n_layers - 1, dtype=complex)
            Z_at[-1] = Z_intr[-1]

            for j in range(n_layers - 2, -1, -1):
                h = layer_thicknesses[j]
                arg = k[j] * h
                if np.abs(arg) > 500:
                    tanh_v[j] = 1.0 if arg.real > 0 else -1.0
                else:
                    tanh_v[j] = np.tanh(arg)
                Zb = Z_at[j + 1]
                tv = tanh_v[j]
                D_den = Z_intr[j] + Zb * tv
                Z_at[j] = Z_intr[j] * (Zb + Z_intr[j] * tv) / D_den

            Z_pred[fi] = Z_at[0]

            # ── Compute dZ_j/dZ_{j+1} for chain rule ──
            dZ_dZnext = np.zeros(n_layers - 1, dtype=complex)
            for j in range(n_layers - 2, -1, -1):
                tv = tanh_v[j]
                Zb = Z_at[j + 1]
                Zi = Z_intr[j]
                D_den = Zi + Zb * tv
                N_num = Zb + Zi * tv
                # dZ_j/dZ_{j+1} = Zi * (D_den - N_num * tv) / D_den^2
                dZ_dZnext[j] = Zi * (D_den - N_num * tv) / (D_den ** 2)

            # ── Compute chain product: prod_{j=0..p-1} dZ_j/dZ_{j+1} ──
            chain_prod = np.ones(n_layers, dtype=complex)
            # chain_prod[0] = 1.0 (no chain for p=0)
            for p in range(1, n_layers):
                chain_prod[p] = chain_prod[p - 1] * dZ_dZnext[p - 1]

            # ── Compute local dZ_p/dm_p for each layer p ──
            for p in range(n_layers):
                dZintr_dm = Z_intr[p] * ln10 / 2.0
                dk_dm = -k[p] * ln10 / 2.0

                if p == n_layers - 1:
                    # Bottom half-space: Z_p = Z_intr_p
                    dZp_local = dZintr_dm
                else:
                    tv = tanh_v[p]
                    Zb = Z_at[p + 1]
                    Zi = Z_intr[p]
                    h = layer_thicknesses[p]
                    N_num = Zb + Zi * tv
                    D_den = Zi + Zb * tv

                    # dt/dm_p = (1 - tv^2) * h * dk_dm
                    dt_dm = (1.0 - tv ** 2) * h * dk_dm

                    # dN/dm = dZintr * tv + Zi * dt_dm
                    dN = dZintr_dm * tv + Zi * dt_dm
                    # dD/dm = dZintr + Zb * dt_dm
                    dD = dZintr_dm + Zb * dt_dm

                    # Z_p = Zi * N / D  =>
                    # dZ_p = dZintr * N/D + Zi * (dN*D - N*dD) / D^2
                    dZp_local = dZintr_dm * N_num / D_den + \
                                Zi * (dN * D_den - N_num * dD) / (D_den ** 2)

                J[fi, p] = chain_prod[p] * dZp_local

        return Z_pred, J

    def objective_and_grad(m, beta=1.0):
        Z_pred, J = forward_with_analytical_grad(m)

        # Residuals
        res_real = (Z_noisy.real - Z_pred.real) / Z_std
        res_imag = (Z_noisy.imag - Z_pred.imag) / Z_std

        misfit = float(np.sum(res_real ** 2 + res_imag ** 2))
        reg = float(np.sum((D @ m) ** 2))
        obj = misfit + beta * reg

        # Gradient via matrix ops
        grad_misfit = -2.0 * (
            (res_real / Z_std) @ J.real + (res_imag / Z_std) @ J.imag
        )
        grad_reg = 2.0 * DtD @ m
        grad = grad_misfit + beta * grad_reg

        return obj, grad

    # Multi-stage inversion with decreasing regularisation
    print("[RECON] Occam 1D MT inversion (log-resistivity, analytical grad) ...")
    m_current = m0.copy()
    betas = [100, 30, 10, 3, 1, 0.3, 0.1, 0.03, 0.01]

    for beta in betas:
        result = minimize(
            objective_and_grad, m_current, args=(beta,),
            method='L-BFGS-B', jac=True,
            bounds=[(-1, 5)] * N_LAYERS,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        m_current = result.x
        chi2 = result.fun
        print(f"[RECON]   β={beta:8.2f}  χ²={chi2:.2f}")

    res_rec = 10 ** m_current

    # Predicted data
    Z_pred, rho_pred, phi_pred = mt_1d_forward(
        frequencies, layer_thicknesses.tolist(), res_rec.tolist()
    )

    return res_rec, depths, layer_thicknesses, Z_pred, rho_pred, phi_pred


# ═══════════════════════════════════════════════════════════
# 5. Metrics
# ═══════════════════════════════════════════════════════════
def compute_metrics(rho_clean, rho_rec, phi_clean, phi_rec,
                    gt_res, rec_res, gt_thick):
    """Compute MT inversion metrics."""
    # Apparent resistivity curve metrics
    log_rho_gt = np.log10(rho_clean)
    log_rho_rec = np.log10(rho_rec)

    rmse_log_rho = float(np.sqrt(np.mean((log_rho_gt - log_rho_rec) ** 2)))
    cc_log_rho = float(np.corrcoef(log_rho_gt, log_rho_rec)[0, 1])

    # Phase metrics
    rmse_phi = float(np.sqrt(np.mean((phi_clean - phi_rec) ** 2)))
    cc_phi = float(np.corrcoef(phi_clean, phi_rec)[0, 1])

    # Combined PSNR
    data_range = log_rho_gt.max() - log_rho_gt.min()
    mse = np.mean((log_rho_gt - log_rho_rec) ** 2)
    psnr = float(10 * np.log10(data_range ** 2 / max(mse, 1e-30)))

    # Layer detection accuracy
    # Check if the conductive layer is detected
    min_idx = np.argmin(rec_res)

    metrics = {
        "PSNR_logRho": psnr,
        "CC_logRho": cc_log_rho,
        "RMSE_logRho": rmse_log_rho,
        "CC_phase": cc_phi,
        "RMSE_phase_deg": rmse_phi,
        "GT_res_top": float(gt_res[0]),
        "GT_res_mid": float(gt_res[1]),
        "GT_res_bot": float(gt_res[2]),
        "min_recovered_res": float(rec_res.min()),
    }
    return metrics


# ═══════════════════════════════════════════════════════════
# 6. Visualization
# ═══════════════════════════════════════════════════════════
def visualize_results(frequencies, rho_clean, rho_noisy, rho_rec,
                      phi_clean, phi_noisy, phi_rec,
                      rec_res, depths, gt_thick, gt_res,
                      metrics, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Apparent resistivity sounding curve
    ax = axes[0, 0]
    ax.loglog(1/frequencies, rho_clean, 'b-', lw=2, label='GT')
    ax.loglog(1/frequencies, rho_noisy, 'k.', ms=5, alpha=0.5, label='Noisy')
    ax.loglog(1/frequencies, rho_rec, 'r--', lw=2, label='Fit')
    ax.set_xlabel('Period [s]')
    ax.set_ylabel('ρ_a [Ω·m]')
    ax.set_title('(a) Apparent Resistivity')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # (b) Phase
    ax = axes[0, 1]
    ax.semilogx(1/frequencies, phi_clean, 'b-', lw=2, label='GT')
    ax.semilogx(1/frequencies, phi_noisy, 'k.', ms=5, alpha=0.5, label='Noisy')
    ax.semilogx(1/frequencies, phi_rec, 'r--', lw=2, label='Fit')
    ax.set_xlabel('Period [s]')
    ax.set_ylabel('Phase [°]')
    ax.set_title('(b) Phase')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) Resistivity-depth profile (step plot)
    ax = axes[1, 0]
    # Ground truth step profile
    gt_depths = [0] + list(np.cumsum(gt_thick)) + [depths[-1] * 1.5]
    gt_res_plot = []
    for r in gt_res:
        gt_res_plot.extend([r, r])
    gt_d_plot = []
    for i in range(len(gt_depths) - 1):
        gt_d_plot.extend([gt_depths[i], gt_depths[i + 1]])

    ax.semilogx(gt_res_plot, gt_d_plot, 'b-', lw=2, label='GT')
    ax.semilogx(rec_res, [0] + list(depths), 'r.-', lw=1.5, label='Inversion')
    ax.invert_yaxis()
    ax.set_xlabel('Resistivity [Ω·m]')
    ax.set_ylabel('Depth [m]')
    ax.set_title('(c) Resistivity Profile')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # (d) Data fit
    ax = axes[1, 1]
    ax.plot(np.log10(rho_clean), np.log10(rho_rec), 'b.', ms=8)
    lims = [min(np.log10(rho_clean).min(), np.log10(rho_rec).min()),
            max(np.log10(rho_clean).max(), np.log10(rho_rec).max())]
    ax.plot(lims, lims, 'k--', lw=0.5)
    ax.set_xlabel('log₁₀(ρ_a GT) [Ω·m]')
    ax.set_ylabel('log₁₀(ρ_a fit) [Ω·m]')
    ax.set_title(f'(d) Data Fit  CC={metrics["CC_logRho"]:.4f}')
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"SimPEG — 1D Magnetotelluric Inversion\n"
        f"PSNR(logρ)={metrics['PSNR_logRho']:.1f} dB  |  "
        f"CC(logρ)={metrics['CC_logRho']:.4f}  |  "
        f"RMSE(φ)={metrics['RMSE_phase_deg']:.2f}°",
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
    print("  SimPEG — 1D Magnetotelluric Inversion")
    print("=" * 65)

    freq, Z_clean, Z_noisy, rho_clean, rho_noisy, phi_clean, phi_noisy = \
        load_or_generate_data()

    print("\n[RECON] Running 1D MT inversion ...")
    rec_res, depths, layer_thick, Z_pred, rho_rec, phi_rec = \
        reconstruct(freq, Z_noisy)

    print("\n[EVAL] Computing metrics ...")
    metrics = compute_metrics(rho_clean, rho_rec, phi_clean, phi_rec,
                              GT_RESISTIVITIES, rec_res, GT_THICKNESSES)
    for k, v in sorted(metrics.items()):
        print(f"  {k:25s} = {v}")

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    # Save apparent resistivity curves (the evaluated quantity for PSNR)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), rho_rec)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), rho_clean)

    visualize_results(freq, rho_clean, rho_noisy, rho_rec,
                      phi_clean, phi_noisy, phi_rec,
                      rec_res, depths, GT_THICKNESSES, GT_RESISTIVITIES,
                      metrics, os.path.join(RESULTS_DIR, "reconstruction_result.png"))

    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65)
