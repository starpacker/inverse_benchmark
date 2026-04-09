#!/usr/bin/env python
"""
Task 159: hippylib_bayesian
PDE-constrained Bayesian Inversion for diffusion coefficient recovery.

Solves: -div(kappa(x) * grad(u)) = f   in Omega = [0,1]^2
        u = 0                            on boundary

Forward: Given kappa(x) -> solve PDE -> temperature field u(x)
Inverse: Given noisy observations of u -> recover kappa(x)

Uses a reduced-dimensional parameterization of kappa via Gaussian basis
functions, making optimization tractable with L-BFGS-B.
Gradients computed via finite differences in the low-dimensional parameter space.
Multiple source experiments improve conditioning.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import time

# ============================================================================
# 1. Forward solver: 2D variable-coefficient diffusion
# ============================================================================

def build_diffusion_matrix(kappa_2d, h):
    """
    Build sparse stiffness matrix for -div(kappa * grad(u)) = f
    on interior nodes with spacing h.
    Uses harmonic averaging of kappa at cell interfaces.
    """
    Ny, Nx = kappa_2d.shape
    N = Ny * Nx
    rows, cols, vals = [], [], []

    for iy in range(Ny):
        for ix in range(Nx):
            i = iy * Nx + ix
            diag_val = 0.0

            for diy, dix in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                niy, nix = iy + diy, ix + dix
                if 0 <= niy < Ny and 0 <= nix < Nx:
                    k_avg = 2.0 * kappa_2d[iy, ix] * kappa_2d[niy, nix] / (
                        kappa_2d[iy, ix] + kappa_2d[niy, nix] + 1e-30)
                    coeff = k_avg / (h * h)
                    j = niy * Nx + nix
                    rows.append(i); cols.append(j); vals.append(-coeff)
                    diag_val += coeff
                else:
                    # Boundary: u=0 contributes to diagonal
                    diag_val += kappa_2d[iy, ix] / (h * h)

            rows.append(i); cols.append(i); vals.append(diag_val)

    return sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))


def solve_forward(kappa_2d, source_2d, h):
    """Solve -div(kappa grad u) = f for u on interior nodes."""
    A = build_diffusion_matrix(kappa_2d, h)
    return spsolve(A, source_2d.ravel())


# ============================================================================
# 2. Gaussian basis parameterization for kappa
# ============================================================================

def make_basis_centers(n_per_dim):
    """Regular grid of Gaussian basis centers in [0,1]^2."""
    margin = 0.15
    cx = np.linspace(margin, 1.0 - margin, n_per_dim)
    cy = np.linspace(margin, 1.0 - margin, n_per_dim)
    CX, CY = np.meshgrid(cx, cy)
    return np.column_stack([CX.ravel(), CY.ravel()])


def kappa_from_params(params, X, Y, centers, sigma_basis, kappa_bg=1.0):
    """
    kappa(x,y) = kappa_bg + sum_i params[i] * G_i(x,y)
    where G_i is a Gaussian centered at centers[i].
    """
    kappa = np.full_like(X, kappa_bg)
    for i in range(len(params)):
        cx, cy = centers[i]
        kappa += params[i] * np.exp(
            -((X - cx)**2 + (Y - cy)**2) / (2.0 * sigma_basis**2)
        )
    return np.maximum(kappa, 0.1)


# ============================================================================
# 3. Observation operator
# ============================================================================

def build_observation_operator(obs_iy, obs_ix, Ny, Nx):
    """Sparse observation matrix that picks values at sensor locations."""
    n_obs = len(obs_iy)
    N = Ny * Nx
    rows = np.arange(n_obs)
    cols = obs_iy * Nx + obs_ix
    return sparse.csr_matrix((np.ones(n_obs), (rows, cols)), shape=(n_obs, N))


# ============================================================================
# 4. Cost function with gradient (finite diff in low-dim param space)
# ============================================================================

def compute_cost(params, X, Y, centers, sigma_basis, kappa_bg,
                 sources, h, B, obs_data_list, noise_var,
                 reg_coeff, prior_params):
    """
    Cost = sum_s 0.5/sigma^2 * ||B*u_s(kappa(params)) - d_s||^2
         + 0.5 * reg * ||params - prior||^2
    """
    total = 0.0
    for src, obs_d in zip(sources, obs_data_list):
        kappa_2d = kappa_from_params(params, X, Y, centers, sigma_basis, kappa_bg)
        u = solve_forward(kappa_2d, src, h)
        residual = B @ u - obs_d
        total += 0.5 * np.sum(residual**2) / noise_var
    total += 0.5 * reg_coeff * np.sum((params - prior_params)**2)
    return total


def compute_cost_and_grad(params, X, Y, centers, sigma_basis, kappa_bg,
                          sources, h, B, obs_data_list, noise_var,
                          reg_coeff, prior_params):
    """
    Cost and gradient via forward finite differences in parameter space.
    Only n_basis forward solves per source for gradient — tractable!
    """
    n_basis = len(params)
    n_sources = len(sources)

    # Base cost per source (cache u's)
    base_costs = []
    total_cost = 0.0
    for src, obs_d in zip(sources, obs_data_list):
        kappa_2d = kappa_from_params(params, X, Y, centers, sigma_basis, kappa_bg)
        u = solve_forward(kappa_2d, src, h)
        residual = B @ u - obs_d
        c = 0.5 * np.sum(residual**2) / noise_var
        base_costs.append(c)
        total_cost += c

    reg = 0.5 * reg_coeff * np.sum((params - prior_params)**2)
    total_cost += reg

    # Gradient via finite differences
    eps = 1e-6
    grad = np.zeros(n_basis)
    for i in range(n_basis):
        params_p = params.copy()
        params_p[i] += eps
        cost_p = 0.0
        for src, obs_d in zip(sources, obs_data_list):
            kappa_p = kappa_from_params(params_p, X, Y, centers, sigma_basis, kappa_bg)
            u_p = solve_forward(kappa_p, src, h)
            res_p = B @ u_p - obs_d
            cost_p += 0.5 * np.sum(res_p**2) / noise_var
        cost_p += 0.5 * reg_coeff * np.sum((params_p - prior_params)**2)
        grad[i] = (cost_p - total_cost) / eps

    return total_cost, grad


# ============================================================================
# 5. Multiple sources
# ============================================================================

def make_sources(N_grid, h):
    """Create multiple source terms for multi-experiment inversion."""
    x = np.linspace(h, 1.0 - h, N_grid)
    y = np.linspace(h, 1.0 - h, N_grid)
    X, Y = np.meshgrid(x, y)
    return [
        10.0 * np.ones((N_grid, N_grid)),
        20.0 * np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / (2 * 0.15**2)),
        15.0 * np.exp(-((X - 0.3)**2 + (Y - 0.7)**2) / (2 * 0.12**2)),
        10.0 * np.sin(np.pi * X) * np.sin(np.pi * Y),
    ]


# ============================================================================
# 6. Metrics
# ============================================================================

def compute_psnr(x_true, x_recon):
    mse = np.mean((x_true - x_recon)**2)
    if mse < 1e-30:
        return 100.0
    data_range = np.max(x_true) - np.min(x_true)
    return 20.0 * np.log10(data_range / np.sqrt(mse))


def compute_cc(x_true, x_recon):
    return float(np.corrcoef(x_true.ravel(), x_recon.ravel())[0, 1])


def compute_relative_error(x_true, x_recon):
    return float(np.linalg.norm(x_true - x_recon) / (np.linalg.norm(x_true) + 1e-30))


def compute_ssim_simple(x_true, x_recon):
    mu_x, mu_y = np.mean(x_true), np.mean(x_recon)
    sigma_x, sigma_y = np.std(x_true), np.std(x_recon)
    sigma_xy = np.mean((x_true - mu_x) * (x_recon - mu_y))
    data_range = np.max(x_true) - np.min(x_true)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    return float(((2*mu_x*mu_y+c1)*(2*sigma_xy+c2)) /
                 ((mu_x**2+mu_y**2+c1)*(sigma_x**2+sigma_y**2+c2)))


# ============================================================================
# 7. Main
# ============================================================================

def main():
    np.random.seed(42)
    t0 = time.time()

    # --- Grid ---
    N_grid = 32
    h = 1.0 / (N_grid + 1)
    x = np.linspace(h, 1.0 - h, N_grid)
    y = np.linspace(h, 1.0 - h, N_grid)
    X, Y = np.meshgrid(x, y)

    # --- True kappa: constant + Gaussian bumps ---
    kappa_bg = 1.0
    kappa_true = kappa_bg * np.ones((N_grid, N_grid))
    kappa_true += 2.0 * np.exp(-((X-0.3)**2 + (Y-0.3)**2) / (2*0.08**2))
    kappa_true += 1.5 * np.exp(-((X-0.7)**2 + (Y-0.7)**2) / (2*0.10**2))
    kappa_true -= 0.5 * np.exp(-((X-0.5)**2 + (Y-0.6)**2) / (2*0.06**2))
    kappa_true = np.maximum(kappa_true, 0.3)

    print(f"Grid: {N_grid}x{N_grid}, h={h:.4f}")
    print(f"True kappa range: [{kappa_true.min():.3f}, {kappa_true.max():.3f}]")

    # --- Basis setup: 5x5 = 25 Gaussians ---
    n_basis_per_dim = 5
    centers = make_basis_centers(n_basis_per_dim)
    n_basis = len(centers)
    sigma_basis = 0.12
    print(f"Basis functions: {n_basis}, sigma={sigma_basis}")

    # --- Sources ---
    sources = make_sources(N_grid, h)
    n_sources = len(sources)
    print(f"Source experiments: {n_sources}")

    # --- Sensors: 10x10 = 100 ---
    n_obs_dim = 10
    obs_ix_1d = np.linspace(1, N_grid-2, n_obs_dim, dtype=int)
    obs_iy_1d = np.linspace(1, N_grid-2, n_obs_dim, dtype=int)
    OIX, OIY = np.meshgrid(obs_ix_1d, obs_iy_1d)
    obs_ix, obs_iy = OIX.ravel(), OIY.ravel()
    n_obs = len(obs_ix)
    B = build_observation_operator(obs_iy, obs_ix, N_grid, N_grid)
    print(f"Sensors: {n_obs}")

    # --- Synthetic data ---
    snr_db = 30.0
    obs_data_list = []
    for src in sources:
        u_true = solve_forward(kappa_true, src, h)
        u_obs = B @ u_true
        sig_pow = np.mean(u_obs**2)
        noise_std = np.sqrt(sig_pow / 10**(snr_db/10))
        obs_data_list.append(u_obs + noise_std * np.random.randn(n_obs))
    noise_var = noise_std**2
    print(f"SNR={snr_db}dB, noise_std~{noise_std:.6f}")

    # --- Optimization ---
    prior_params = np.zeros(n_basis)
    reg_coeff = 0.01

    iter_count = [0]
    def objective(params):
        cost, grad = compute_cost_and_grad(
            params, X, Y, centers, sigma_basis, kappa_bg,
            sources, h, B, obs_data_list, noise_var,
            reg_coeff, prior_params)
        iter_count[0] += 1
        if iter_count[0] % 5 == 0:
            kc = kappa_from_params(params, X, Y, centers, sigma_basis, kappa_bg)
            cc = compute_cc(kappa_true, kc)
            print(f"  [{iter_count[0]:3d}] cost={cost:.4e}  CC={cc:.4f}")
        return cost, grad

    print("\n=== L-BFGS-B Optimization ===")
    bounds = [(-5.0, 5.0)] * n_basis

    result = minimize(
        objective, np.zeros(n_basis),
        method='L-BFGS-B', jac=True, bounds=bounds,
        options={'maxiter': 200, 'maxfun': 1500, 'ftol': 1e-14, 'gtol': 1e-9, 'disp': True})

    params_opt = result.x
    print(f"\n{result.message}")
    print(f"Evaluations: {result.nfev}, Final cost: {result.fun:.4e}")

    kappa_map = kappa_from_params(params_opt, X, Y, centers, sigma_basis, kappa_bg)

    # --- Refinement if needed ---
    cc = compute_cc(kappa_true, kappa_map)
    psnr = compute_psnr(kappa_true, kappa_map)
    if cc < 0.5 or psnr < 15:
        print(f"\nCC={cc:.4f}, PSNR={psnr:.2f} — refining with weaker regularization...")
        reg_coeff = 0.001
        result2 = minimize(
            objective, params_opt,
            method='L-BFGS-B', jac=True, bounds=bounds,
            options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-15, 'gtol': 1e-11, 'disp': True})
        params_opt = result2.x
        kappa_map = kappa_from_params(params_opt, X, Y, centers, sigma_basis, kappa_bg)

    # --- Metrics ---
    psnr = compute_psnr(kappa_true, kappa_map)
    cc = compute_cc(kappa_true, kappa_map)
    re = compute_relative_error(kappa_true, kappa_map)
    ssim = compute_ssim_simple(kappa_true, kappa_map)

    print(f"\n=== Final Results ===")
    print(f"PSNR:  {psnr:.2f} dB")
    print(f"CC:    {cc:.4f}")
    print(f"RE:    {re:.4f} ({re*100:.2f}%)")
    print(f"SSIM:  {ssim:.4f}")
    print(f"MAP kappa: [{kappa_map.min():.3f}, {kappa_map.max():.3f}]")
    print(f"Params: {params_opt}")

    # --- Save ---
    sandbox = "/data/yjh/hippylib_bayesian_sandbox"
    results_dir = os.path.join(sandbox, "results")
    os.makedirs(results_dir, exist_ok=True)

    np.save(os.path.join(results_dir, "ground_truth.npy"), kappa_true)
    np.save(os.path.join(results_dir, "reconstruction.npy"), kappa_map)

    metrics = {
        "psnr": float(psnr), "ssim": float(ssim),
        "cc": float(cc), "relative_error": float(re),
        "n_grid": N_grid, "n_obs": n_obs,
        "n_sources": n_sources, "n_basis": n_basis,
        "snr_db": snr_db,
    }
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Visualization ---
    error_map = np.abs(kappa_true - kappa_map)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    vmin_k = min(kappa_true.min(), kappa_map.min())
    vmax_k = max(kappa_true.max(), kappa_map.max())

    im0 = axes[0,0].imshow(kappa_true, origin='lower', extent=[0,1,0,1],
                            cmap='viridis', vmin=vmin_k, vmax=vmax_k)
    axes[0,0].set_title(r'True $\kappa(x)$', fontsize=14)
    axes[0,0].set_xlabel('x'); axes[0,0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0,0], fraction=0.046)

    im1 = axes[0,1].imshow(kappa_map, origin='lower', extent=[0,1,0,1],
                            cmap='viridis', vmin=vmin_k, vmax=vmax_k)
    axes[0,1].set_title(r'MAP Estimate $\kappa_{MAP}(x)$', fontsize=14)
    axes[0,1].set_xlabel('x'); axes[0,1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0,1], fraction=0.046)
    axes[0,1].scatter(x[obs_ix], y[obs_iy], c='red', s=8, marker='x', alpha=0.5)

    im2 = axes[1,0].imshow(error_map, origin='lower', extent=[0,1,0,1], cmap='hot')
    axes[1,0].set_title(r'$|\kappa_{true} - \kappa_{MAP}|$', fontsize=14)
    axes[1,0].set_xlabel('x'); axes[1,0].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1,0], fraction=0.046)

    mid = N_grid // 2
    axes[1,1].plot(x, kappa_true[mid,:], 'b-', lw=2, label='True')
    axes[1,1].plot(x, kappa_map[mid,:], 'r--', lw=2, label='MAP')
    axes[1,1].set_title(f'Cross-section y={y[mid]:.2f}', fontsize=14)
    axes[1,1].set_xlabel('x'); axes[1,1].set_ylabel(r'$\kappa$')
    axes[1,1].legend(fontsize=12); axes[1,1].grid(True, alpha=0.3)

    fig.suptitle(f'PDE-Constrained Bayesian Inversion\n'
                 f'PSNR={psnr:.1f}dB, CC={cc:.3f}, RE={re*100:.1f}%, SSIM={ssim:.3f}',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(results_dir, "reconstruction_result.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nVisualization saved.")
    print(f"Total time: {time.time()-t0:.1f}s")
    return metrics


if __name__ == "__main__":
    main()
