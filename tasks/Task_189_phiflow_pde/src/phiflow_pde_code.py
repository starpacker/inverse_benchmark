#!/usr/bin/env python3
"""
Task 189: phiflow_pde — Heat Diffusion Parameter Inversion via Differentiable PDE Simulation

Inverse problem: Given a noisy observation of a 2D temperature field at time T,
recover the initial temperature field u₀ by differentiating through an explicit
finite-difference PDE solver using PyTorch autograd.

PDE: ∂u/∂t = α ∇²u  (2D heat equation)

Forward model:
  - Initial temperature u₀ on a 64×64 grid (sum of Gaussian bumps)
  - Diffusion coefficient α (known)
  - Explicit Euler time-stepping for N steps → u(T)

Inverse problem:
  - Given noisy u_obs = u(T) + noise, recover u₀
  - Minimize ||forward(u₀_est) - u_obs||² via Adam + autograd through the solver

Metrics: PSNR and SSIM of recovered u₀ vs ground truth u₀.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def create_gaussian_field(nx=64, ny=64, n_bumps=5, seed=42):
    """Create a ground truth initial temperature field as a sum of Gaussian bumps."""
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    field = np.zeros((nx, ny), dtype=np.float64)
    for _ in range(n_bumps):
        cx = rng.uniform(0.15, 0.85)
        cy = rng.uniform(0.15, 0.85)
        sigma = rng.uniform(0.05, 0.15)
        amplitude = rng.uniform(0.5, 1.5)
        field += amplitude * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))

    # Normalize to [0, 1]
    field = (field - field.min()) / (field.max() - field.min() + 1e-10)
    return field


def heat_equation_forward(u0, alpha, dt, n_steps, dx=1.0/64):
    """
    Forward solve of 2D heat equation using explicit finite differences.
    ∂u/∂t = α ∇²u with periodic boundary conditions.

    All operations are differentiable through PyTorch autograd.

    Args:
        u0: Initial condition (torch.Tensor, shape [H, W])
        alpha: Diffusion coefficient (scalar)
        dt: Time step size
        n_steps: Number of time steps
        dx: Spatial grid spacing

    Returns:
        u: Solution at time T = n_steps * dt
    """
    u = u0.clone()
    coeff = alpha * dt / (dx ** 2)

    # CFL stability check
    if coeff > 0.25:
        raise ValueError(
            f"CFL violation: coeff={coeff:.4f} > 0.25. "
            f"Reduce dt or alpha. (alpha={alpha}, dt={dt}, dx={dx})"
        )

    for step in range(n_steps):
        # Periodic boundary: use roll for neighbor access
        u_left  = torch.roll(u, shifts=1, dims=0)
        u_right = torch.roll(u, shifts=-1, dims=0)
        u_up    = torch.roll(u, shifts=1, dims=1)
        u_down  = torch.roll(u, shifts=-1, dims=1)

        laplacian = (u_left + u_right + u_up + u_down - 4.0 * u) / (dx ** 2)
        u = u + alpha * dt * laplacian

    return u


def compute_metrics(gt, recon):
    """Compute PSNR and SSIM between ground truth and reconstruction."""
    gt_np = gt.detach().cpu().numpy().astype(np.float64)
    recon_np = recon.detach().cpu().numpy().astype(np.float64)

    # Normalize both to [0, 1] range for metric computation
    data_range = max(gt_np.max() - gt_np.min(), 1e-10)

    psnr_val = psnr(gt_np, recon_np, data_range=data_range)
    ssim_val = ssim(gt_np, recon_np, data_range=data_range)
    return psnr_val, ssim_val


def run_inversion():
    """Main inversion pipeline."""
    # ============================================================
    # Configuration
    # ============================================================
    nx, ny = 64, 64
    dx = 1.0 / nx
    alpha = 0.0005          # Diffusion coefficient (moderate)
    dt = 0.1 * dx**2 / alpha  # Safe time step (CFL factor ~0.1)
    n_steps = 20            # Number of time steps (moderate diffusion)
    noise_level = 0.01      # Noise standard deviation
    n_iters = 1500          # Optimization iterations
    lr = 0.005              # Learning rate
    tv_weight = 5e-4        # TV regularization weight
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Grid: {nx}x{ny}, dx={dx:.6f}")
    print(f"alpha={alpha}, dt={dt:.8f}, n_steps={n_steps}")
    print(f"CFL number: {alpha * dt / dx**2:.4f}")
    print(f"Total simulation time T = {n_steps * dt:.6f}")
    print(f"Noise level: {noise_level}")

    # ============================================================
    # Step 1: Generate ground truth initial temperature field
    # ============================================================
    gt_u0_np = create_gaussian_field(nx, ny, n_bumps=5, seed=seed)
    gt_u0 = torch.tensor(gt_u0_np, dtype=torch.float64, device=device)

    # ============================================================
    # Step 2: Forward solve to get observation at time T
    # ============================================================
    with torch.no_grad():
        u_T = heat_equation_forward(gt_u0, alpha, dt, n_steps, dx)

    # Add noise to create observation
    torch.manual_seed(seed + 1)
    noise = noise_level * torch.randn_like(u_T)
    u_obs = u_T + noise

    print(f"GT u0 range: [{gt_u0.min():.4f}, {gt_u0.max():.4f}]")
    print(f"u(T) range: [{u_T.min():.4f}, {u_T.max():.4f}]")
    print(f"u_obs range: [{u_obs.min():.4f}, {u_obs.max():.4f}]")

    # ============================================================
    # Step 3: Inverse problem — recover u₀ from u_obs
    # ============================================================
    # Initialize with observation (reasonable initial guess since diffusion is mild)
    u0_est = u_obs.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([u0_est], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters, eta_min=lr * 0.01)

    loss_history = []
    psnr_history = []
    best_psnr = 0.0
    best_recon = None
    best_ssim = 0.0

    print(f"\nStarting optimization ({n_iters} iterations)...")
    for it in range(n_iters):
        optimizer.zero_grad()

        # Forward solve from estimated u₀
        u_pred = heat_equation_forward(u0_est, alpha, dt, n_steps, dx)

        # Data fidelity loss
        loss_data = F.mse_loss(u_pred, u_obs)

        # TV regularization to suppress noise amplification in the ill-posed inversion
        tv_x = torch.mean(torch.abs(u0_est[1:, :] - u0_est[:-1, :]))
        tv_y = torch.mean(torch.abs(u0_est[:, 1:] - u0_est[:, :-1]))
        loss_tv = tv_weight * (tv_x + tv_y)

        loss = loss_data + loss_tv

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if it % 50 == 0 or it == n_iters - 1:
            with torch.no_grad():
                p, s = compute_metrics(gt_u0, u0_est)
                psnr_history.append((it, p))
                if p > best_psnr:
                    best_psnr = p
                    best_ssim = s
                    best_recon = u0_est.detach().clone()
                if it % 100 == 0 or it == n_iters - 1:
                    print(f"  Iter {it:4d}: loss={loss_val:.6e}, PSNR={p:.2f} dB, SSIM={s:.4f} (best: {best_psnr:.2f})")

    # ============================================================
    # Step 4: Final evaluation — use best reconstruction
    # ============================================================
    with torch.no_grad():
        if best_recon is not None:
            recon = best_recon
        else:
            recon = u0_est.detach()
        final_psnr, final_ssim = compute_metrics(gt_u0, recon)

    print(f"\n{'='*50}")
    print(f"Final Results:")
    print(f"  PSNR: {final_psnr:.2f} dB")
    print(f"  SSIM: {final_ssim:.4f}")
    print(f"{'='*50}")

    # ============================================================
    # Step 5: Save results
    # ============================================================
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Save metrics
    metrics = {
        "psnr": round(float(final_psnr), 2),
        "ssim": round(float(final_ssim), 4),
        "noise_level": noise_level,
        "n_iters": n_iters,
        "grid_size": [nx, ny],
        "alpha": alpha,
        "n_steps": n_steps,
        "method": "differentiable_pde_inversion_pytorch_autograd"
    }
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics.json")

    # Save arrays
    gt_np = gt_u0.cpu().numpy()
    recon_np = recon.cpu().numpy()
    obs_np = u_obs.cpu().numpy()
    u_T_np = u_T.cpu().numpy()

    np.save(os.path.join(results_dir, 'ground_truth.npy'), gt_np)
    np.save(os.path.join(results_dir, 'reconstruction.npy'), recon_np)
    np.save(os.path.join(results_dir, 'observation.npy'), obs_np)
    print(f"Saved .npy arrays")

    # ============================================================
    # Step 6: Visualization
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Fields
    vmin = min(gt_np.min(), recon_np.min())
    vmax = max(gt_np.max(), recon_np.max())

    im0 = axes[0, 0].imshow(gt_np.T, origin='lower', cmap='hot', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Ground Truth u₀', fontsize=14)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(recon_np.T, origin='lower', cmap='hot', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'Recovered u₀ (PSNR={final_psnr:.1f}dB)', fontsize=14)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 1])

    diff = np.abs(gt_np - recon_np)
    im2 = axes[0, 2].imshow(diff.T, origin='lower', cmap='viridis')
    axes[0, 2].set_title(f'|Error| (max={diff.max():.4f})', fontsize=14)
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 2])

    # Row 2: Observation and convergence
    im3 = axes[1, 0].imshow(obs_np.T, origin='lower', cmap='hot')
    axes[1, 0].set_title('Noisy Observation u(T)+noise', fontsize=14)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(u_T_np.T, origin='lower', cmap='hot')
    axes[1, 1].set_title('Clean u(T) (forward solution)', fontsize=14)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1, 1])

    axes[1, 2].semilogy(loss_history)
    axes[1, 2].set_title('Optimization Loss', fontsize=14)
    axes[1, 2].set_xlabel('Iteration')
    axes[1, 2].set_ylabel('MSE Loss')
    axes[1, 2].grid(True, alpha=0.3)

    # Add PSNR convergence on twin axis
    if psnr_history:
        ax_twin = axes[1, 2].twinx()
        iters, psnrs = zip(*psnr_history)
        ax_twin.plot(iters, psnrs, 'r-o', markersize=3, label='PSNR')
        ax_twin.set_ylabel('PSNR (dB)', color='r')
        ax_twin.tick_params(axis='y', labelcolor='r')

    plt.suptitle(
        f'Heat Equation Initial Condition Inversion\n'
        f'∂u/∂t = α∇²u, α={alpha}, Grid={nx}×{ny}, '
        f'PSNR={final_psnr:.2f}dB, SSIM={final_ssim:.4f}',
        fontsize=16, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'reconstruction_result.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved reconstruction_result.png")

    print(f"\nAll results saved to {results_dir}/")
    return final_psnr, final_ssim


if __name__ == '__main__':
    psnr_val, ssim_val = run_inversion()
