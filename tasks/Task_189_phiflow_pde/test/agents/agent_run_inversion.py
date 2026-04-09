import numpy as np

import torch

import torch.nn.functional as F

import matplotlib

matplotlib.use('Agg')

from skimage.metrics import structural_similarity as ssim

from skimage.metrics import peak_signal_noise_ratio as psnr

def forward_operator(u0, alpha, dt, n_steps, dx):
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
        u: Solution at time T = n_steps * dt (torch.Tensor)
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
        u_left = torch.roll(u, shifts=1, dims=0)
        u_right = torch.roll(u, shifts=-1, dims=0)
        u_up = torch.roll(u, shifts=1, dims=1)
        u_down = torch.roll(u, shifts=-1, dims=1)
        
        laplacian = (u_left + u_right + u_up + u_down - 4.0 * u) / (dx ** 2)
        u = u + alpha * dt * laplacian
    
    return u

def run_inversion(data_dict, n_iters=1500, lr=0.005, tv_weight=5e-4):
    """
    Run the inverse problem to recover initial condition u₀ from noisy observation.
    
    Minimizes ||forward(u₀_est) - u_obs||² + TV(u₀_est) via Adam optimizer
    with autograd through the PDE solver.
    
    Args:
        data_dict: Dictionary from load_and_preprocess_data containing:
            - gt_u0: Ground truth initial condition
            - u_obs: Noisy observation
            - params: Simulation parameters
        n_iters: Number of optimization iterations
        lr: Learning rate
        tv_weight: Total variation regularization weight
        
    Returns:
        result_dict: Dictionary containing:
            - best_recon: Best reconstruction (torch.Tensor)
            - final_recon: Final reconstruction (torch.Tensor)
            - loss_history: List of loss values
            - psnr_history: List of (iteration, psnr) tuples
            - best_psnr: Best PSNR achieved
            - best_ssim: SSIM at best PSNR
    """
    gt_u0 = data_dict['gt_u0']
    u_obs = data_dict['u_obs']
    params = data_dict['params']
    
    alpha = params['alpha']
    dt = params['dt']
    n_steps = params['n_steps']
    dx = params['dx']
    
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
        u_pred = forward_operator(u0_est, alpha, dt, n_steps, dx)
        
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
                gt_np = gt_u0.detach().cpu().numpy().astype(np.float64)
                recon_np = u0_est.detach().cpu().numpy().astype(np.float64)
                data_range = max(gt_np.max() - gt_np.min(), 1e-10)
                p = psnr(gt_np, recon_np, data_range=data_range)
                s = ssim(gt_np, recon_np, data_range=data_range)
                
                psnr_history.append((it, p))
                if p > best_psnr:
                    best_psnr = p
                    best_ssim = s
                    best_recon = u0_est.detach().clone()
                if it % 100 == 0 or it == n_iters - 1:
                    print(f"  Iter {it:4d}: loss={loss_val:.6e}, PSNR={p:.2f} dB, SSIM={s:.4f} (best: {best_psnr:.2f})")
    
    result_dict = {
        'best_recon': best_recon if best_recon is not None else u0_est.detach(),
        'final_recon': u0_est.detach(),
        'loss_history': loss_history,
        'psnr_history': psnr_history,
        'best_psnr': best_psnr,
        'best_ssim': best_ssim
    }
    
    return result_dict
