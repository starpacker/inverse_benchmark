import numpy as np

import matplotlib

matplotlib.use('Agg')

def load_and_preprocess_data(nx, ny, lx, ly, rho, noise_level, seed):
    """
    Load and preprocess data: Generate Taylor-Green vortex velocity/pressure fields
    and add PIV measurement noise to velocity.
    
    Parameters:
        nx, ny: Grid resolution
        lx, ly: Domain size [m]
        rho: Fluid density [kg/m³]
        noise_level: Velocity noise σ / U_max
        seed: Random seed
    
    Returns:
        data_dict: Dictionary containing all velocity, pressure, and grid data
    """
    rng = np.random.default_rng(seed)
    
    # Create grid
    x = np.linspace(0, lx, nx, endpoint=False)
    y = np.linspace(0, ly, ny, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    # Taylor-Green vortex analytic solution
    k = 2 * np.pi / lx
    U = 1.0
    
    u_gt = U * np.cos(k * xx) * np.sin(k * yy)
    v_gt = -U * np.sin(k * xx) * np.cos(k * yy)
    p_gt = -(rho * U**2 / 4) * (np.cos(2 * k * xx) + np.cos(2 * k * yy))
    
    # Add PIV noise
    U_scale = max(np.abs(u_gt).max(), np.abs(v_gt).max())
    sigma = noise_level * U_scale
    u_noisy = u_gt + sigma * rng.standard_normal(u_gt.shape)
    v_noisy = v_gt + sigma * rng.standard_normal(v_gt.shape)
    
    # Compute grid spacing
    dx = lx / nx
    dy = ly / ny
    
    data_dict = {
        'u_gt': u_gt,
        'v_gt': v_gt,
        'p_gt': p_gt,
        'u_noisy': u_noisy,
        'v_noisy': v_noisy,
        'xx': xx,
        'yy': yy,
        'dx': dx,
        'dy': dy,
        'rho': rho,
        'nx': nx,
        'ny': ny,
        'lx': lx,
        'ly': ly
    }
    
    return data_dict
