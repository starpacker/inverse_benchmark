import matplotlib

matplotlib.use('Agg')

import numpy as np

def sphere_form_factor_amplitude(q, R):
    """
    Normalised form-factor amplitude for a homogeneous sphere:
        f(q,R) = 3 [sin(qR) - qR cos(qR)] / (qR)^3
    Returns f(q,R).  P(q) = f^2.
    """
    qR = np.asarray(q * R, dtype=np.float64)
    result = np.ones_like(qR)
    mask = qR > 1e-12
    result[mask] = 3.0 * (np.sin(qR[mask]) - qR[mask] * np.cos(qR[mask])) / qR[mask]**3
    return result

def sphere_intensity(q, R, scale, background):
    """
    I(q) = scale * V * delta_rho^2 * P(q) + background
    where P(q) = |f(q,R)|^2 and V = (4/3)pi R^3.
    
    For simplicity we fold V*delta_rho^2 into the scale factor
    so:  I(q) = scale * P(q) + background
    This is the standard parameterisation used by SasView.
    """
    P_q = sphere_form_factor_amplitude(q, R)**2
    return scale * P_q + background

def forward_operator(q, params):
    """
    Forward operator: Compute scattering intensity I(q) from model parameters.
    
    The forward model is:
        I(q) = scale * P(q, R) + background
    where P(q) = |f(q,R)|^2 is the form factor of a sphere.
    
    Parameters:
    -----------
    q : ndarray
        Scattering vector values (Å^-1)
    params : dict
        Model parameters with keys 'radius', 'scale', 'background'
        
    Returns:
    --------
    I_pred : ndarray
        Predicted scattering intensity
    """
    print("[FORWARD] Forward model: I(q) = scale * P(q,R) + background")
    print("[FORWARD] P(q) = [3(sin(qR)-qR cos(qR))/(qR)^3]^2")
    
    R = params['radius']
    scale = params['scale']
    background = params['background']
    
    I_pred = sphere_intensity(q, R, scale, background)
    
    return I_pred
