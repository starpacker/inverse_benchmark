import numpy as np

import matplotlib

matplotlib.use('Agg')

def generate_faraday_depth_spectrum(phi_arr, components, d_phi):
    """
    Generate ground truth Faraday depth spectrum F(φ) on a given φ grid.
    Each component is a delta function convolved with a narrow Gaussian
    for numerical representation.
    """
    F_gt = np.zeros(len(phi_arr), dtype=complex)
    
    for comp in components:
        phi0 = comp['phi']
        amp = comp['amplitude']
        chi0 = comp['chi0']
        
        # Represent as narrow Gaussian (delta-like)
        sigma_phi = d_phi * 0.5
        gaussian = amp * np.exp(-(phi_arr - phi0)**2 / (2 * sigma_phi**2))
        phase = np.exp(2j * chi0)
        F_gt += gaussian * phase
    
    return F_gt
