import matplotlib

matplotlib.use('Agg')

import numpy as np

from scipy.special import voigt_profile

def gaussian(x, amplitude, center, sigma):
    """Gaussian peak."""
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)

def lorentzian(x, amplitude, center, gamma):
    """Lorentzian peak."""
    return amplitude * gamma**2 / ((x - center)**2 + gamma**2)

def voigt(x, amplitude, center, sigma, gamma):
    """Voigt peak – convolution of Gaussian and Lorentzian."""
    vp = voigt_profile(x - center, sigma, gamma)
    vp_max = voigt_profile(0.0, sigma, gamma)
    if vp_max > 0:
        return amplitude * vp / vp_max
    return np.zeros_like(x)

def forward_operator(x, params, peak_defs):
    """
    Evaluate the composite spectral model given parameters.
    
    This is the forward model: parameters -> predicted spectrum
    
    Args:
        x: numpy array of x-axis values
        params: lmfit Parameters object or dict-like with parameter values
        peak_defs: list of dicts defining peak types
    
    Returns:
        y_pred: numpy array of predicted spectrum values
    """
    # Get baseline parameters
    if hasattr(params, '__getitem__'):
        # lmfit Parameters object
        baseline_intercept = params['baseline_intercept'].value if hasattr(params['baseline_intercept'], 'value') else params['baseline_intercept']
        baseline_slope = params['baseline_slope'].value if hasattr(params['baseline_slope'], 'value') else params['baseline_slope']
    else:
        baseline_intercept = params['baseline_intercept']
        baseline_slope = params['baseline_slope']
    
    y = baseline_intercept + baseline_slope * x
    
    for i, pk in enumerate(peak_defs):
        if hasattr(params, '__getitem__'):
            amp_param = params[f'p{i}_amplitude']
            cen_param = params[f'p{i}_center']
            amp = amp_param.value if hasattr(amp_param, 'value') else amp_param
            cen = cen_param.value if hasattr(cen_param, 'value') else cen_param
        else:
            amp = params[f'p{i}_amplitude']
            cen = params[f'p{i}_center']
        
        if pk["type"] == "gaussian":
            if hasattr(params, '__getitem__'):
                sig_param = params[f'p{i}_sigma']
                sig = sig_param.value if hasattr(sig_param, 'value') else sig_param
            else:
                sig = params[f'p{i}_sigma']
            y = y + gaussian(x, amp, cen, sig)
            
        elif pk["type"] == "lorentzian":
            if hasattr(params, '__getitem__'):
                gam_param = params[f'p{i}_gamma']
                gam = gam_param.value if hasattr(gam_param, 'value') else gam_param
            else:
                gam = params[f'p{i}_gamma']
            y = y + lorentzian(x, amp, cen, gam)
            
        elif pk["type"] == "voigt":
            if hasattr(params, '__getitem__'):
                sig_param = params[f'p{i}_sigma']
                gam_param = params[f'p{i}_gamma']
                sig = sig_param.value if hasattr(sig_param, 'value') else sig_param
                gam = gam_param.value if hasattr(gam_param, 'value') else gam_param
            else:
                sig = params[f'p{i}_sigma']
                gam = params[f'p{i}_gamma']
            y = y + voigt(x, amp, cen, sig, gam)
    
    return y
