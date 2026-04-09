import numpy as np

import matplotlib

matplotlib.use('Agg')

import warnings

warnings.filterwarnings('ignore')

def load_and_preprocess_data(nx, ny, dx, xmin, xmax, ymin, ymax,
                              true_beta, true_zt, true_dz, true_c,
                              noise_level, seed=42):
    """
    Generate a synthetic magnetic anomaly grid whose radial power
    spectrum follows the Bouligand et al. (2009) model.
    
    Returns:
        grid_noisy: Noisy magnetic anomaly grid
        grid_clean: Clean magnetic anomaly grid
        params: Dictionary with grid and true parameters
    """
    from pycurious import bouligand2009
    
    np.random.seed(seed)
    
    # Create wavenumber grids
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi  # rad/m
    ky = np.fft.fftfreq(ny, d=dx) * 2 * np.pi  # rad/m
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    kh_grid = np.sqrt(kx_grid**2 + ky_grid**2)
    
    # Avoid k=0
    kh_grid[0, 0] = 1e-10
    
    # Convert wavenumber to rad/km for bouligand2009
    kh_km = kh_grid * 1000.0  # rad/km
    
    # Compute log power spectrum from true parameters
    log_phi = bouligand2009(kh_km, true_beta, true_zt, true_dz, true_c)
    phi = np.exp(log_phi)
    
    # Create random phase grid
    phase = 2 * np.pi * np.random.rand(ny, nx)
    
    # Construct Fourier coefficients with correct power and random phase
    amplitude = np.sqrt(phi)
    fourier_coeff = amplitude * np.exp(1j * phase)
    
    # DC component real
    fourier_coeff[0, 0] = np.abs(fourier_coeff[0, 0])
    
    # Inverse FFT to get spatial domain
    grid_clean = np.real(np.fft.ifft2(fourier_coeff))
    
    # Add noise
    noise = noise_level * np.std(grid_clean) * np.random.randn(ny, nx)
    grid_noisy = grid_clean + noise
    
    true_curie = true_zt + true_dz
    print(f"  Grid shape: {grid_noisy.shape}")
    print(f"  Anomaly range: [{grid_noisy.min():.1f}, {grid_noisy.max():.1f}] nT")
    print(f"  Grid extent: {xmin/1e3:.0f}-{xmax/1e3:.0f} km x {ymin/1e3:.0f}-{ymax/1e3:.0f} km")
    print(f"  True params: beta={true_beta:.2f}, zt={true_zt:.2f} km, dz={true_dz:.2f} km")
    print(f"  True Curie depth: {true_curie:.2f} km")
    
    params = {
        'nx': nx, 'ny': ny, 'dx': dx,
        'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax,
        'true_beta': true_beta, 'true_zt': true_zt, 'true_dz': true_dz, 'true_c': true_c,
        'true_curie_depth': true_curie,
        'noise_level': noise_level
    }
    
    return grid_noisy, grid_clean, params
