import numpy as np


# --- Extracted Dependencies ---

def get_absorption_spectra(wavelengths):
    """
    Get absorption spectra for Hb and HbO2 at given wavelengths.
    Returns interpolated values based on reference data.
    """
    ref_wavelengths = np.array([700, 730, 760, 800, 850, 900])
    hb_ref = np.array([100, 80, 60, 40, 30, 20])
    hbo2_ref = np.array([30, 40, 50, 60, 70, 80])
    
    from scipy.interpolate import interp1d
    
    hb_interp = interp1d(ref_wavelengths, hb_ref, kind='linear', 
                         bounds_error=False, fill_value='extrapolate')
    hbo2_interp = interp1d(ref_wavelengths, hbo2_ref, kind='linear',
                           bounds_error=False, fill_value='extrapolate')
    
    hb = hb_interp(wavelengths)
    hbo2 = hbo2_interp(wavelengths)
    
    return hb, hbo2

def perform_spectral_unmixing(reconstruction, wavelengths):
    """
    Perform linear spectral unmixing to estimate Hb and HbO2 concentrations.
    
    Args:
        reconstruction: Reconstructed images, shape (n_wl, nz, ny, nx)
        wavelengths: Array of wavelengths used
        
    Returns:
        concentrations: Array of shape (2, nz, ny, nx) where [0] is Hb, [1] is HbO2
    """
    hb, hbo2 = get_absorption_spectra(wavelengths)
    
    E = np.vstack([hb, hbo2]).T
    
    print("Spectral Unmixing...")
    n_wl, nz, ny, nx = reconstruction.shape
    S = reconstruction.reshape(n_wl, -1)
    
    E_inv = np.linalg.pinv(E)
    C = E_inv @ S
    
    concentrations = C.reshape(2, nz, ny, nx)
    
    return concentrations
