import numpy as np

import matplotlib

matplotlib.use('Agg')

import warnings

warnings.filterwarnings('ignore')

def load_and_preprocess_data(nx, ny, pixel_size, noise_level, seed=42):
    """
    Generate synthetic convergence map with cluster-like structures,
    compute true shear field, and add noise.
    
    Returns:
        kappa_true: True convergence map
        g1_obs: Observed shear component 1 (with noise)
        g2_obs: Observed shear component 2 (with noise)
        g1_true: True shear component 1
        g2_true: True shear component 2
        params: Dictionary of parameters
    """
    np.random.seed(seed)
    
    # Generate synthetic convergence map
    x = np.arange(nx) - nx / 2
    y = np.arange(ny) - ny / 2
    X, Y = np.meshgrid(x, y)
    
    kappa = np.zeros((ny, nx))
    
    # Add several NFW-like mass concentrations (clusters)
    clusters = [
        {'x0': 20, 'y0': 30, 'amp': 0.35, 'rs': 15},
        {'x0': -40, 'y0': -20, 'amp': 0.28, 'rs': 12},
        {'x0': 60, 'y0': -50, 'amp': 0.20, 'rs': 10},
        {'x0': -70, 'y0': 60, 'amp': 0.25, 'rs': 14},
        {'x0': 10, 'y0': -80, 'amp': 0.15, 'rs': 8},
    ]
    
    for c in clusters:
        r = np.sqrt((X - c['x0'])**2 + (Y - c['y0'])**2)
        rs = c['rs']
        x_nfw = r / rs
        kappa_nfw = np.zeros_like(r)
        mask = x_nfw > 0.01
        kappa_nfw[mask] = c['amp'] / (x_nfw[mask]**2 - 1 + 1e-10) * (
            1 - 1 / np.sqrt(np.abs(x_nfw[mask]**2 - 1) + 1e-10) *
            np.where(x_nfw[mask] < 1,
                     np.arctan(np.sqrt(np.abs(1 - x_nfw[mask]**2) + 1e-10)),
                     np.arctanh(np.sqrt(np.abs(x_nfw[mask]**2 - 1) /
                                        (x_nfw[mask]**2 + 1e-10))))
        )
        kappa += np.clip(kappa_nfw, 0, c['amp'] * 3)
    
    # Add filamentary structures
    theta_fil = np.deg2rad(30)
    d_fil = (X * np.sin(theta_fil) - Y * np.cos(theta_fil))
    kappa += 0.08 * np.exp(-d_fil**2 / (2 * 8**2))
    
    theta_fil2 = np.deg2rad(-45)
    d_fil2 = (X * np.sin(theta_fil2) - Y * np.cos(theta_fil2))
    kappa += 0.06 * np.exp(-d_fil2**2 / (2 * 6**2))
    
    # Smooth with Gaussian
    from scipy.ndimage import gaussian_filter
    kappa = gaussian_filter(kappa, sigma=2.0)
    
    # Ensure non-negative
    kappa_true = np.clip(kappa, 0, None)
    
    print(f"  Convergence map shape: {kappa_true.shape}")
    print(f"  κ range: [{kappa_true.min():.4f}, {kappa_true.max():.4f}]")
    print(f"  {len(clusters)} cluster halos + 2 filaments")
    
    # Compute true shear using forward operator
    g1_true, g2_true = forward_operator(kappa_true)
    
    print(f"[FWD] γ1 range: [{g1_true.min():.4f}, {g1_true.max():.4f}]")
    print(f"[FWD] γ2 range: [{g2_true.min():.4f}, {g2_true.max():.4f}]")
    
    # Add shape noise
    g1_obs = g1_true + noise_level * np.random.randn(ny, nx)
    g2_obs = g2_true + noise_level * np.random.randn(ny, nx)
    print(f"[DATA] Added Gaussian shape noise σ={noise_level}")
    
    params = {
        'nx': nx,
        'ny': ny,
        'pixel_size': pixel_size,
        'noise_level': noise_level,
    }
    
    return kappa_true, g1_obs, g2_obs, g1_true, g2_true, params

def forward_operator(kappa):
    """
    Forward: Compute shear field from convergence map using Kaiser-Squires.
    
    γ(k) = D(k) · κ(k) where D is the Kaiser-Squires kernel.
    
    The KS kernel in Fourier space is:
    D(k) = (k1² - k2² + 2i·k1·k2) / (k1² + k2²)
    
    Args:
        kappa: Convergence map (2D array)
    
    Returns:
        g1: Shear component 1
        g2: Shear component 2
    """
    ny, nx = kappa.shape
    
    # Create frequency grids
    k1 = np.fft.fftfreq(nx)
    k2 = np.fft.fftfreq(ny)
    K1, K2 = np.meshgrid(k1, k2)
    
    # Compute k² avoiding division by zero
    k_sq = K1**2 + K2**2
    k_sq[0, 0] = 1.0  # Avoid division by zero at DC
    
    # Kaiser-Squires kernel components
    # D1 = (k1² - k2²) / k²
    # D2 = 2·k1·k2 / k²
    D1 = (K1**2 - K2**2) / k_sq
    D2 = 2 * K1 * K2 / k_sq
    
    # Set DC component to zero
    D1[0, 0] = 0.0
    D2[0, 0] = 0.0
    
    # Forward transform of convergence
    kappa_fft = np.fft.fft2(kappa)
    
    # Apply KS kernel to get shear in Fourier space
    g1_fft = D1 * kappa_fft
    g2_fft = D2 * kappa_fft
    
    # Inverse transform to get shear in real space
    g1 = np.real(np.fft.ifft2(g1_fft))
    g2 = np.real(np.fft.ifft2(g2_fft))
    
    return g1, g2
