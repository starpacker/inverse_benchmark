import numpy as np

import torch

import matplotlib

matplotlib.use('Agg')

def load_and_preprocess_data(nx=64, ny=64, n_bumps=5, alpha=0.0005, n_steps=20, 
                              noise_level=0.01, seed=42, device=None):
    """
    Load and preprocess data for the heat equation inversion problem.
    
    Creates ground truth initial temperature field, runs forward simulation,
    and generates noisy observation.
    
    Args:
        nx, ny: Grid dimensions
        n_bumps: Number of Gaussian bumps in initial field
        alpha: Diffusion coefficient
        n_steps: Number of time steps for forward simulation
        noise_level: Standard deviation of observation noise
        seed: Random seed
        device: PyTorch device
        
    Returns:
        data_dict: Dictionary containing:
            - gt_u0: Ground truth initial condition (torch.Tensor)
            - u_obs: Noisy observation at time T (torch.Tensor)
            - u_T: Clean solution at time T (torch.Tensor)
            - params: Dictionary of simulation parameters
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dx = 1.0 / nx
    dt = 0.1 * dx**2 / alpha  # Safe time step (CFL factor ~0.1)
    
    # Create ground truth initial temperature field as sum of Gaussian bumps
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
    
    gt_u0 = torch.tensor(field, dtype=torch.float64, device=device)
    
    # Forward solve to get observation at time T
    with torch.no_grad():
        u = gt_u0.clone()
        coeff = alpha * dt / (dx ** 2)
        
        if coeff > 0.25:
            raise ValueError(
                f"CFL violation: coeff={coeff:.4f} > 0.25. "
                f"Reduce dt or alpha."
            )
        
        for step in range(n_steps):
            u_left = torch.roll(u, shifts=1, dims=0)
            u_right = torch.roll(u, shifts=-1, dims=0)
            u_up = torch.roll(u, shifts=1, dims=1)
            u_down = torch.roll(u, shifts=-1, dims=1)
            
            laplacian = (u_left + u_right + u_up + u_down - 4.0 * u) / (dx ** 2)
            u = u + alpha * dt * laplacian
        
        u_T = u.clone()
    
    # Add noise to create observation
    torch.manual_seed(seed + 1)
    noise = noise_level * torch.randn_like(u_T)
    u_obs = u_T + noise
    
    params = {
        'nx': nx,
        'ny': ny,
        'dx': dx,
        'dt': dt,
        'alpha': alpha,
        'n_steps': n_steps,
        'noise_level': noise_level,
        'device': device
    }
    
    print(f"Device: {device}")
    print(f"Grid: {nx}x{ny}, dx={dx:.6f}")
    print(f"alpha={alpha}, dt={dt:.8f}, n_steps={n_steps}")
    print(f"CFL number: {alpha * dt / dx**2:.4f}")
    print(f"Total simulation time T = {n_steps * dt:.6f}")
    print(f"Noise level: {noise_level}")
    print(f"GT u0 range: [{gt_u0.min():.4f}, {gt_u0.max():.4f}]")
    print(f"u(T) range: [{u_T.min():.4f}, {u_T.max():.4f}]")
    print(f"u_obs range: [{u_obs.min():.4f}, {u_obs.max():.4f}]")
    
    data_dict = {
        'gt_u0': gt_u0,
        'u_obs': u_obs,
        'u_T': u_T,
        'params': params
    }
    
    return data_dict
