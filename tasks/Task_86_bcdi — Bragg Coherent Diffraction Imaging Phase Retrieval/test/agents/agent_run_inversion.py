import numpy as np

import matplotlib

matplotlib.use('Agg')

import warnings

warnings.filterwarnings('ignore')

def run_inversion(intensity, support, n_hio=200, n_er=50, n_cycles=5, beta=0.9):
    """
    Phase retrieval using Hybrid Input-Output (HIO) + Error Reduction (ER).
    
    HIO update:
        Inside support:  ρ_{n+1} = P_F(ρ_n)
        Outside support: ρ_{n+1} = ρ_n - β * P_F(ρ_n)
    
    ER update:
        ρ_{n+1} = P_S(P_F(ρ_n))
    
    where P_F = Fourier constraint (replace amplitude, keep phase)
          P_S = Support constraint (zero outside support)
    
    Args:
        intensity: 3D measured diffraction intensity
        support: 3D binary support mask
        n_hio: number of HIO iterations per cycle
        n_er: number of ER iterations per cycle
        n_cycles: number of HIO+ER cycles
        beta: HIO feedback parameter
        
    Returns:
        obj_recon: reconstructed 3D complex object
        errors: list of R-factor values during convergence
    """
    amplitudes = np.sqrt(np.maximum(intensity, 0))
    
    # Initialize with random phases
    phases_init = 2 * np.pi * np.random.rand(*intensity.shape)
    rho = support * np.exp(1j * phases_init)
    
    errors = []
    
    for cycle in range(n_cycles):
        # HIO phase
        for i in range(n_hio):
            # Fourier projection
            rho_ft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(rho)))
            rho_ft_proj = amplitudes * np.exp(1j * np.angle(rho_ft))
            rho_prime = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(rho_ft_proj)))
            
            # HIO update
            rho_new = np.where(support > 0.5, rho_prime, rho - beta * rho_prime)
            rho = rho_new
            
            # Compute error
            if (i + 1) % 50 == 0:
                err = np.sum(np.abs(np.abs(rho_ft) - amplitudes)**2) / np.sum(amplitudes**2)
                errors.append(err)
        
        # ER phase
        for i in range(n_er):
            rho_ft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(rho)))
            rho_ft_proj = amplitudes * np.exp(1j * np.angle(rho_ft))
            rho = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(rho_ft_proj)))
            
            # Support projection
            rho = rho * support
            
            if (i + 1) % 25 == 0:
                err = np.sum(np.abs(np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(rho)))) - amplitudes)**2) / np.sum(amplitudes**2)
                errors.append(err)
        
        print(f"  Cycle {cycle+1}/{n_cycles}: R-factor = {errors[-1]:.6f}")
    
    return rho, errors
