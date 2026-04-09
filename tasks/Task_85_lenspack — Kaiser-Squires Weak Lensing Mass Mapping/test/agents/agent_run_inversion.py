import numpy as np

import matplotlib

matplotlib.use('Agg')

import warnings

warnings.filterwarnings('ignore')

def run_inversion(g1_obs, g2_obs):
    """
    Inverse: Recover convergence from observed shear using Kaiser-Squires.
    
    κ̂(k) = D*(k) · γ̂(k)
    
    where D* is the complex conjugate of the KS kernel.
    
    The E-mode convergence is recovered via:
    κ_E = Re[IFFT( D1* · γ1 + D2* · γ2 )]
    
    The B-mode (should be zero for pure lensing) is:
    κ_B = Re[IFFT( -D2* · γ1 + D1* · γ2 )]
    
    Args:
        g1_obs: Observed shear component 1
        g2_obs: Observed shear component 2
    
    Returns:
        kE: E-mode convergence (mass map)
        kB: B-mode convergence (should be noise-like)
    """
    print("  [KS93] Running Kaiser-Squires inversion...")
    
    ny, nx = g1_obs.shape
    
    # Create frequency grids
    k1 = np.fft.fftfreq(nx)
    k2 = np.fft.fftfreq(ny)
    K1, K2 = np.meshgrid(k1, k2)
    
    # Compute k² avoiding division by zero
    k_sq = K1**2 + K2**2
    k_sq[0, 0] = 1.0  # Avoid division by zero at DC
    
    # Kaiser-Squires kernel components (same as forward)
    D1 = (K1**2 - K2**2) / k_sq
    D2 = 2 * K1 * K2 / k_sq
    
    # Set DC component to zero
    D1[0, 0] = 0.0
    D2[0, 0] = 0.0
    
    # FFT of observed shear
    g1_fft = np.fft.fft2(g1_obs)
    g2_fft = np.fft.fft2(g2_obs)
    
    # Inverse KS: κ_E = D1 · γ1 + D2 · γ2 (using real kernels)
    # κ_B = -D2 · γ1 + D1 · γ2
    kE_fft = D1 * g1_fft + D2 * g2_fft
    kB_fft = -D2 * g1_fft + D1 * g2_fft
    
    # Inverse transform
    kE = np.real(np.fft.ifft2(kE_fft))
    kB = np.real(np.fft.ifft2(kB_fft))
    
    print(f"  [KS93] E-mode κ range: [{kE.min():.4f}, {kE.max():.4f}]")
    print(f"  [KS93] B-mode κ range: [{kB.min():.4f}, {kB.max():.4f}]")
    print(f"  [KS93] B/E ratio: {np.std(kB)/np.std(kE):.4f}")
    
    return kE, kB
