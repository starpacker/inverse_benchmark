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
