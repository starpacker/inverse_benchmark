import numpy as np

import matplotlib

matplotlib.use('Agg')

import warnings

warnings.filterwarnings('ignore')

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
