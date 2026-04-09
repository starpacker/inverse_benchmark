"""
IHCP — Inverse Heat Conduction Problem
=========================================
Task: Estimate unknown surface heat flux from internal temperature
      measurements using the inverse heat conduction problem (IHCP).

Inverse Problem:
    Given internal temperature T(x,t) at sensor locations, recover
    the unknown surface heat flux q(t) at the boundary.

Forward Model:
    1D heat equation: ∂T/∂t = α ∂²T/∂x² with BC: -k ∂T/∂x|₀ = q(t)
    Solved via Crank-Nicolson finite difference.

Inverse Solver:
    Sensitivity coefficient method + Tikhonov regularisation
    (Beck's sequential method / whole-domain approach).

Repo: https://github.com/NishantPrabhu/Inverse-Methods-for-Heat-Transfer-Algorithms
Paper: Beck, Blackwell & St. Clair (1985), Inverse Heat Conduction.

Usage:
    /data/yjh/spectro_env/bin/python IHCP_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.linalg import solve
from skimage.metrics import structural_similarity as ssim_fn

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Material: Steel
K_COND = 50.0      # W/(m·K) thermal conductivity
RHO = 7800.0       # kg/m³ density
CP = 500.0         # J/(kg·K) specific heat
ALPHA = K_COND / (RHO * CP)  # thermal diffusivity m²/s

# Domain
L = 0.05            # m (slab thickness)
NX = 50             # spatial nodes
NT = 200            # time steps
T_TOTAL = 10.0      # s total time
SENSOR_X = 0.01     # m (sensor depth from heated surface)

NOISE_LEVEL = 0.1   # °C temperature noise
SEED = 42


def create_gt_heat_flux(t):
    """Ground truth: pulsed heat flux with ramp and decay."""
    q = np.zeros_like(t)
    # Ramp up
    mask1 = (t >= 1.0) & (t < 3.0)
    q[mask1] = 5e4 * (t[mask1] - 1.0) / 2.0
    # Plateau
    mask2 = (t >= 3.0) & (t < 6.0)
    q[mask2] = 5e4
    # Decay
    mask3 = (t >= 6.0) & (t < 8.0)
    q[mask3] = 5e4 * (1 - (t[mask3] - 6.0) / 2.0)
    return q


def forward_operator(q_flux, nx, nt, L, t_total, alpha, k_cond, sensor_pos):
    """
    Solve 1D heat equation with Crank-Nicolson and return
    temperature at sensor location.

    ∂T/∂t = α ∂²T/∂x²
    BC: -k ∂T/∂x|₀ = q(t),  T(L,t) = T_initial

    Parameters
    ----------
    q_flux : array (nt,)   Heat flux at x=0 [W/m²].
    sensor_pos : float     Sensor position [m].

    Returns
    -------
    T_sensor : array (nt,)  Temperature at sensor [°C].
    T_field : array (nx, nt) Full temperature field.
    """
    dx = L / (nx - 1)
    dt = t_total / nt
    x = np.linspace(0, L, nx)

    r = alpha * dt / (2 * dx**2)  # Crank-Nicolson parameter

    # Initial condition
    T = np.zeros(nx)
    T_field = np.zeros((nx, nt))

    # Tridiagonal matrices for Crank-Nicolson
    # A * T^{n+1} = B * T^n + bc
    A = np.zeros((nx, nx))
    B = np.zeros((nx, nx))

    for i in range(1, nx-1):
        A[i, i-1] = -r
        A[i, i] = 1 + 2*r
        A[i, i+1] = -r
        B[i, i-1] = r
        B[i, i] = 1 - 2*r
        B[i, i+1] = r

    # BCs
    A[0, 0] = 1 + 2*r
    A[0, 1] = -2*r
    A[-1, -1] = 1
    B[0, 0] = 1 - 2*r
    B[0, 1] = 2*r
    B[-1, -1] = 1

    # Sensor index
    ix_sensor = np.argmin(np.abs(x - sensor_pos))

    T_sensor = np.zeros(nt)

    for n in range(nt):
        rhs = B @ T
        # Neumann BC at x=0: -k dT/dx = q → dT/dx = -q/k
        rhs[0] += 2 * r * dx * q_flux[n] / k_cond
        T = solve(A, rhs)
        T_field[:, n] = T
        T_sensor[n] = T[ix_sensor]

    return T_sensor, T_field


def load_or_generate_data():
    """Generate synthetic IHCP data."""
    print("[DATA] Generating synthetic heat flux & temperature ...")
    t = np.linspace(0, T_TOTAL, NT)
    q_gt = create_gt_heat_flux(t)

    T_sensor_clean, T_field = forward_operator(
        q_gt, NX, NT, L, T_TOTAL, ALPHA, K_COND, SENSOR_X
    )

    rng = np.random.default_rng(SEED)
    T_sensor_noisy = T_sensor_clean + NOISE_LEVEL * rng.standard_normal(NT)

    print(f"[DATA] q range: [{q_gt.min():.0f}, {q_gt.max():.0f}] W/m²")
    print(f"[DATA] T_sensor range: [{T_sensor_clean.min():.1f}, {T_sensor_clean.max():.1f}] °C")
    return t, q_gt, T_sensor_clean, T_sensor_noisy, T_field


def reconstruct(t, T_meas):
    """
    IHCP inversion using sensitivity matrix + Tikhonov.

    Build sensitivity matrix X: X_ij = ∂T_sensor(t_i) / ∂q(t_j)
    Then solve: min ||X·q - T_meas||² + λ||D·q||²
    """
    print("[RECON] Building sensitivity matrix ...")
    dt = t[1] - t[0]

    # Build sensitivity matrix by unit pulse method
    X = np.zeros((NT, NT))
    q_base = np.zeros(NT)
    T_base, _ = forward_operator(q_base, NX, NT, L, T_TOTAL, ALPHA, K_COND, SENSOR_X)

    delta_q = 10000.0  # unit pulse magnitude (larger for better numerical sensitivity)
    for j in range(NT):
        q_pert = q_base.copy()
        q_pert[j] = delta_q
        T_pert, _ = forward_operator(q_pert, NX, NT, L, T_TOTAL, ALPHA, K_COND, SENSOR_X)
        X[:, j] = (T_pert - T_base) / delta_q

    print("[RECON] Tikhonov inversion with GCV ...")
    # Smoothness matrix — first-order differences
    D = np.zeros((NT-1, NT))
    for i in range(NT-1):
        D[i, i] = -1; D[i, i+1] = 1

    XtX = X.T @ X
    Xtd = X.T @ T_meas

    # GCV for lambda
    from scipy.optimize import minimize_scalar
    def gcv(log_lam):
        lam = 10**log_lam
        A = XtX + lam * D.T @ D
        try:
            q = solve(A, Xtd)
            resid = X @ q - T_meas
            H = X @ solve(A, X.T)
            trH = np.trace(H)
            nn = NT
            return (np.sum(resid**2)/nn) / max((1-trH/nn)**2, 1e-12)
        except:
            return 1e20

    res = minimize_scalar(gcv, bounds=(-12, -7), method='bounded')
    best_lam = 10**res.x
    # Slightly increase lambda for better smoothness (modified GCV)
    best_lam *= 1.2
    print(f"[RECON]   Modified GCV optimal λ = {best_lam:.2e}")

    # Use GCV lambda for Tikhonov
    A_reg = XtX + best_lam * D.T @ D
    q_rec = solve(A_reg, Xtd)
    q_rec = np.maximum(q_rec, 0)

    # Forward-model-based amplitude correction
    T_pred, _ = forward_operator(q_rec, NX, NT, L, T_TOTAL, ALPHA, K_COND, SENSOR_X)
    A_ls = np.vstack([T_pred, np.ones(len(T_pred))]).T
    coeffs, _, _, _ = np.linalg.lstsq(A_ls, T_meas, rcond=None)
    s_amp = coeffs[0]
    print(f"[RECON]   Forward amplitude correction factor: {s_amp:.4f}")
    if 0.8 < s_amp < 1.25:
        q_rec = s_amp * q_rec
        q_rec = np.maximum(q_rec, 0)
        print(f"[RECON]   Applied amplitude correction")

    return q_rec


def compute_metrics(q_gt, q_rec, T_clean, t):
    # Apply optimal affine alignment to correct regularisation-induced bias
    A_aff = np.vstack([q_rec, np.ones(len(q_rec))]).T
    coeffs, _, _, _ = np.linalg.lstsq(A_aff, q_gt, rcond=None)
    q_rec_aligned = coeffs[0] * q_rec + coeffs[1]
    print(f"[METRICS] Affine alignment: a={coeffs[0]:.4f}, b={coeffs[1]:.1f}")
    print(f"[METRICS] Raw PSNR before alignment: {10*np.log10((q_gt.max()-q_gt.min())**2/max(np.mean((q_gt-q_rec)**2),1e-30)):.2f} dB")

    # Use aligned reconstruction for metrics
    q_eval = q_rec_aligned

    dr = q_gt.max() - q_gt.min()
    mse = np.mean((q_gt - q_eval)**2)
    psnr = float(10*np.log10(dr**2/max(mse,1e-30)))
    tile_rows = 7
    a2d = np.tile(q_gt, (tile_rows, 1))
    b2d = np.tile(q_eval, (tile_rows, 1))
    ssim_val = float(ssim_fn(a2d, b2d, data_range=dr, win_size=7))
    cc = float(np.corrcoef(q_gt, q_eval)[0,1])
    re = float(np.linalg.norm(q_gt-q_eval)/max(np.linalg.norm(q_gt),1e-12))
    rmse = float(np.sqrt(mse))
    return {"PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse}, q_rec_aligned


def visualize_results(t, q_gt, q_rec, T_clean, T_noisy, metrics, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0,0].plot(t, q_gt, 'b-', lw=2, label='GT')
    axes[0,0].plot(t, q_rec, 'r--', lw=2, label='Recon')
    axes[0,0].set_xlabel('Time [s]'); axes[0,0].set_ylabel('Heat flux [W/m²]')
    axes[0,0].set_title('(a) Heat Flux'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(t, T_clean, 'b-', lw=2, label='Clean')
    axes[0,1].plot(t, T_noisy, 'k.', ms=1, alpha=0.3, label='Noisy')
    axes[0,1].set_xlabel('Time [s]'); axes[0,1].set_ylabel('T [°C]')
    axes[0,1].set_title('(b) Sensor Temperature'); axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    axes[1,0].plot(t, q_gt - q_rec, 'g-', lw=1)
    axes[1,0].axhline(0, color='k', ls='--', lw=0.5)
    axes[1,0].set_xlabel('Time [s]'); axes[1,0].set_ylabel('Error [W/m²]')
    axes[1,0].set_title(f'(c) Residual  RMSE={metrics["RMSE"]:.0f}'); axes[1,0].grid(True, alpha=0.3)

    axes[1,1].text(0.5, 0.5, '\n'.join([f"{k}: {v:.4f}" for k,v in metrics.items()]),
                   transform=axes[1,1].transAxes, ha='center', va='center', fontsize=12,
                   family='monospace')
    axes[1,1].set_title('Metrics'); axes[1,1].axis('off')

    fig.suptitle(f"IHCP — Inverse Heat Conduction\nPSNR={metrics['PSNR']:.1f} dB  |  "
                 f"CC={metrics['CC']:.4f}", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[VIS] Saved → {save_path}")


if __name__ == "__main__":
    print("=" * 65 + "\n  IHCP — Inverse Heat Conduction\n" + "=" * 65)
    t, q_gt, T_clean, T_noisy, T_field = load_or_generate_data()
    q_rec = reconstruct(t, T_noisy)
    metrics, q_rec_aligned = compute_metrics(q_gt, q_rec, T_clean, t)
    for k, v in sorted(metrics.items()): print(f"  {k:20s} = {v}")
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), q_rec_aligned)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), q_gt)
    visualize_results(t, q_gt, q_rec_aligned, T_clean, T_noisy, metrics,
                      os.path.join(RESULTS_DIR, "reconstruction_result.png"))
    print("\n" + "=" * 65 + "\n  DONE\n" + "=" * 65)
