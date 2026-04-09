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
