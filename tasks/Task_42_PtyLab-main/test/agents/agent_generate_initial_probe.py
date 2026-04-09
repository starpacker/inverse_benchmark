import numpy as np

from scipy.ndimage import shift, gaussian_filter

def generate_initial_probe(Np, dxo, diameter):
    """Generates a soft-edged disk probe."""
    Y, X = np.meshgrid(np.arange(Np), np.arange(Np), indexing='ij')
    X = X - Np // 2
    Y = Y - Np // 2
    R = np.sqrt(X**2 + Y**2)
    
    if diameter is not None:
         fwhm_pix = diameter / dxo
         radius_pix = fwhm_pix / 2.0
    else:
         radius_pix = Np / 8 
         
    probe = np.zeros((Np, Np), dtype=np.complex128)
    probe[R <= radius_pix] = 1.0
    # Soften edges slightly
    probe = gaussian_filter(probe.real, sigma=2.0) + 0j
    return probe
