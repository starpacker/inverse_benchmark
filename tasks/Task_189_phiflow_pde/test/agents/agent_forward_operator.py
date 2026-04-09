import torch

import matplotlib

matplotlib.use('Agg')

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
