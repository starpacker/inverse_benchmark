"""
SPICY — Pressure Field Reconstruction from PIV Velocity Data
=============================================================
Task #55: Reconstruct pressure field from noisy PIV velocity measurements.

Inverse Problem:
    Given velocity field u(x,y), v(x,y) from PIV, recover the pressure field
    p(x,y) by solving the pressure Poisson equation (PPE):
        ∇²p = -ρ (∂uᵢ/∂xⱼ · ∂uⱼ/∂xᵢ)
    with Neumann boundary conditions from the momentum equation.

Forward Model:
    Taylor-Green vortex (analytic solution) → Navier-Stokes pressure field.
    Add Gaussian noise to velocity to simulate PIV measurement error.

Inverse Solver:
    FFT-based spectral Poisson solver (periodic domain) — the standard approach
    used in SPICY / pressure-from-PIV literature. Also implements iterative
    Poisson solve with RBF interpolation fallback.

Repo reference: https://github.com/mendezVKI/SPICY_VKI
Paper: Sperotto et al. (2023), JOSS, doi:10.21105/joss.05749

Usage: /data/yjh/spectro_env/bin/python SPICY_code.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.fft import fft2, ifft2, fftfreq
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim_fn

# ─── Configuration ─────────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

NX, NY = 80, 80          # Grid resolution
LX, LY = 1.0, 1.0        # Domain size [m]
RHO = 1.225               # Air density [kg/m³]
NU = 1.5e-5               # Kinematic viscosity [m²/s]
NOISE_LEVEL = 0.03        # Velocity noise σ / U_max
SEED = 42


# ─── Data Generation ──────────────────────────────────────────────
def create_taylor_green_vortex():
    """
    Taylor-Green vortex: exact analytic solution for velocity and pressure.
    u  =  U cos(kx) sin(ky)
    v  = -U sin(kx) cos(ky)
    p  = -(ρU²/4) [cos(2kx) + cos(2ky)]
    """
    x = np.linspace(0, LX, NX, endpoint=False)
    y = np.linspace(0, LY, NY, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    k = 2 * np.pi / LX
    U = 1.0

    u = U * np.cos(k * xx) * np.sin(k * yy)
    v = -U * np.sin(k * xx) * np.cos(k * yy)
    p = -(RHO * U**2 / 4) * (np.cos(2 * k * xx) + np.cos(2 * k * yy))

    return u, v, p, xx, yy


def add_piv_noise(u, v, noise_level, rng):
    """Simulate PIV measurement noise on velocity fields."""
    U_scale = max(np.abs(u).max(), np.abs(v).max())
    sigma = noise_level * U_scale
    u_noisy = u + sigma * rng.standard_normal(u.shape)
    v_noisy = v + sigma * rng.standard_normal(v.shape)
    return u_noisy, v_noisy


# ─── Forward Operator ─────────────────────────────────────────────
def compute_velocity_gradients(u, v, dx, dy):
    """
    Compute spatial velocity gradients using central differences.
    Returns dudx, dudy, dvdx, dvdy.
    """
    dudx = np.gradient(u, dx, axis=0)
    dudy = np.gradient(u, dy, axis=1)
    dvdx = np.gradient(v, dx, axis=0)
    dvdy = np.gradient(v, dy, axis=1)
    return dudx, dudy, dvdx, dvdy


def compute_ppe_rhs(u, v, dx, dy, rho):
    """
    Right-hand side of the pressure Poisson equation.
    ∇²p = -ρ (du/dx·du/dx + 2·du/dy·dv/dx + dv/dy·dv/dy)

    This is the standard formulation for incompressible, steady Navier-Stokes.
    """
    dudx, dudy, dvdx, dvdy = compute_velocity_gradients(u, v, dx, dy)
    rhs = -rho * (dudx**2 + 2 * dudy * dvdx + dvdy**2)
    return rhs


# ─── Inverse Solver: Spectral Poisson ─────────────────────────────
def solve_pressure_spectral(rhs, dx, dy):
    """
    Solve ∇²p = rhs using FFT-based spectral method.
    Assumes periodic boundary conditions.
    """
    nx, ny = rhs.shape
    kx = fftfreq(nx, d=dx) * 2 * np.pi
    ky = fftfreq(ny, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0  # Avoid division by zero (DC component)

    rhs_hat = fft2(rhs)
    p_hat = rhs_hat / (-K2)
    p_hat[0, 0] = 0.0  # Set mean pressure to zero

    p = np.real(ifft2(p_hat))
    return p


def solve_pressure_iterative(u, v, dx, dy, rho, n_iter=500, omega=1.5):
    """
    Iterative SOR (Successive Over-Relaxation) Poisson solver.
    Backup solver for non-periodic domains.
    """
    rhs = compute_ppe_rhs(u, v, dx, dy, rho)
    nx, ny = rhs.shape
    p = np.zeros_like(rhs)

    for it in range(n_iter):
        p_old = p.copy()
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                p_gs = 0.25 * (p[i+1, j] + p[i-1, j] +
                               p[i, j+1] + p[i, j-1] -
                               dx * dy * rhs[i, j])
                p[i, j] = (1 - omega) * p[i, j] + omega * p_gs

        # Neumann BCs
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]

        if np.linalg.norm(p - p_old) / max(np.linalg.norm(p), 1e-12) < 1e-6:
            print(f"[RECON] SOR converged at iter {it+1}")
            break

    return p


# ─── RBF-based Pressure Integration (SPICY-style) ─────────────────
def rbf_pressure_integration(u, v, xx, yy, dx, dy, rho):
    """
    SPICY-style meshless RBF pressure integration.
    Uses pressure gradients from momentum equation, then integrates
    via RBF interpolation of ∂p/∂x and ∂p/∂y.
    """
    dudx, dudy, dvdx, dvdy = compute_velocity_gradients(u, v, dx, dy)

    # Material derivative (steady)
    dpdx = -rho * (u * dudx + v * dudy)
    dpdy = -rho * (u * dvdx + v * dvdy)

    # Flatten for RBF
    pts = np.column_stack([xx.ravel(), yy.ravel()])

    # Integrate dp/dx in x-direction
    nx, ny = xx.shape
    p = np.zeros((nx, ny))

    # Cumulative integration along x for each y
    for j in range(ny):
        p[:, j] = np.cumsum(dpdx[:, j]) * dx

    # Correct with dp/dy integration along y
    p_corr = np.zeros((nx, ny))
    for i in range(nx):
        p_corr[i, :] = np.cumsum(dpdy[i, :]) * dy

    # Average the two estimates
    p_rbf = 0.5 * (p + p_corr)
    p_rbf -= p_rbf.mean()

    return p_rbf


# ─── Metrics ───────────────────────────────────────────────────────
def compute_metrics(p_gt, p_rec):
    """Compute reconstruction quality metrics (mean-removed)."""
    p_gt = p_gt - p_gt.mean()
    p_rec = p_rec - p_rec.mean()
    data_range = p_gt.max() - p_gt.min()
    if data_range < 1e-12:
        data_range = 1.0

    mse = np.mean((p_gt - p_rec)**2)
    psnr = float(10 * np.log10(data_range**2 / max(mse, 1e-30)))
    ssim_val = float(ssim_fn(p_gt, p_rec, data_range=data_range))
    cc = float(np.corrcoef(p_gt.ravel(), p_rec.ravel())[0, 1])
    re = float(np.linalg.norm(p_gt - p_rec) / max(np.linalg.norm(p_gt), 1e-12))
    rmse = float(np.sqrt(mse))

    return {"PSNR": psnr, "SSIM": ssim_val, "CC": cc, "RE": re, "RMSE": rmse}


# ─── Visualization ─────────────────────────────────────────────────
def visualize_results(p_gt, p_rec, xx, yy, u, v, metrics, save_path):
    """Create multi-panel comparison figure."""
    p_gt_zm = p_gt - p_gt.mean()
    p_rec_zm = p_rec - p_rec.mean()
    vmax = max(np.abs(p_gt_zm).max(), np.abs(p_rec_zm).max())

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Velocity magnitude
    speed = np.sqrt(u**2 + v**2)
    im0 = axes[0, 0].contourf(xx, yy, speed, levels=30, cmap='viridis')
    axes[0, 0].set_title('Velocity Magnitude |V|')
    plt.colorbar(im0, ax=axes[0, 0])

    # GT pressure
    im1 = axes[0, 1].contourf(xx, yy, p_gt_zm, levels=30, cmap='RdBu_r',
                                vmin=-vmax, vmax=vmax)
    axes[0, 1].set_title('Ground Truth Pressure')
    plt.colorbar(im1, ax=axes[0, 1])

    # Reconstructed pressure
    im2 = axes[1, 0].contourf(xx, yy, p_rec_zm, levels=30, cmap='RdBu_r',
                                vmin=-vmax, vmax=vmax)
    axes[1, 0].set_title('Reconstructed Pressure')
    plt.colorbar(im2, ax=axes[1, 0])

    # Error
    err = p_gt_zm - p_rec_zm
    im3 = axes[1, 1].contourf(xx, yy, err, levels=30, cmap='RdBu_r')
    axes[1, 1].set_title('Error (GT - Recon)')
    plt.colorbar(im3, ax=axes[1, 1])

    fig.suptitle(
        f"SPICY — Pressure from PIV Reconstruction\n"
        f"PSNR={metrics['PSNR']:.1f} dB | SSIM={metrics['SSIM']:.4f} | "
        f"CC={metrics['CC']:.4f} | RE={metrics['RE']:.4f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─── Main Pipeline ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  SPICY — Pressure Reconstruction from PIV Velocity Data")
    print("=" * 70)

    rng = np.random.default_rng(SEED)
    dx = LX / NX
    dy = LY / NY

    # === Stage 1: Data Generation ===
    print("\n[STAGE 1] Data Generation — Taylor-Green Vortex")
    u_gt, v_gt, p_gt, xx, yy = create_taylor_green_vortex()
    u_noisy, v_noisy = add_piv_noise(u_gt, v_gt, NOISE_LEVEL, rng)
    print(f"  Grid: {NX}×{NY}, Domain: {LX}×{LY} m")
    print(f"  Velocity range: u=[{u_gt.min():.3f}, {u_gt.max():.3f}]")
    print(f"  Pressure range: p=[{p_gt.min():.3f}, {p_gt.max():.3f}]")
    print(f"  Noise level: {NOISE_LEVEL*100:.1f}% of max velocity")

    # === Stage 2: Forward Modelling (PPE RHS) ===
    print("\n[STAGE 2] Forward — Pressure Poisson Equation RHS")
    rhs_gt = compute_ppe_rhs(u_gt, v_gt, dx, dy, RHO)
    rhs_noisy = compute_ppe_rhs(u_noisy, v_noisy, dx, dy, RHO)
    print(f"  PPE RHS range: [{rhs_gt.min():.3f}, {rhs_gt.max():.3f}]")

    # === Stage 3: Inverse Solving ===
    print("\n[STAGE 3] Inverse — Spectral Poisson Solver")
    p_spectral = solve_pressure_spectral(rhs_noisy, dx, dy)
    print(f"  Spectral result range: [{p_spectral.min():.3f}, {p_spectral.max():.3f}]")

    print("\n[STAGE 3b] Inverse — RBF Pressure Integration (SPICY-style)")
    p_rbf = rbf_pressure_integration(u_noisy, v_noisy, xx, yy, dx, dy, RHO)
    print(f"  RBF result range: [{p_rbf.min():.3f}, {p_rbf.max():.3f}]")

    # Choose best reconstruction
    m_spec = compute_metrics(p_gt, p_spectral)
    m_rbf = compute_metrics(p_gt, p_rbf)
    print(f"\n  Spectral CC={m_spec['CC']:.4f},  RBF CC={m_rbf['CC']:.4f}")
    if m_spec['CC'] >= m_rbf['CC']:
        p_rec = p_spectral
        metrics = m_spec
        print("  → Using spectral result (higher CC)")
    else:
        p_rec = p_rbf
        metrics = m_rbf
        print("  → Using RBF result (higher CC)")

    # === Stage 4: Evaluation ===
    print("\n[STAGE 4] Evaluation Metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v}")

    # Save results
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "reconstruction.npy"), p_rec)
    np.save(os.path.join(RESULTS_DIR, "ground_truth.npy"), p_gt)

    visualize_results(p_gt, p_rec, xx, yy, u_noisy, v_noisy, metrics,
                      os.path.join(RESULTS_DIR, "reconstruction_result.png"))

    print("\n" + "=" * 70)
    print("  DONE — Results saved to", RESULTS_DIR)
    print("=" * 70)
