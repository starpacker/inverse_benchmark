import numpy as np

import matplotlib

matplotlib.use('Agg')

def harvey_comp(freq, zeta, nc):
    """Single Harvey component: P(ν) = ζ / (1 + (ν/ν_c)²)"""
    return zeta / (1.0 + (freq / nc) ** 2)

def bg_model(freq, z1, nc1, z2, nc2, w):
    """Background = 2 Harvey components + white noise."""
    return harvey_comp(freq, z1, nc1) + harvey_comp(freq, z2, nc2) + w

def osc_modes(freq, numax, dnu, sigma_env, height, width):
    """Lorentzian modes modulated by Gaussian envelope."""
    eps = 1.5
    modes = np.zeros_like(freq)
    n_lo = int(np.floor((numax - 4 * sigma_env) / dnu))
    n_hi = int(np.ceil((numax + 4 * sigma_env) / dnu))

    for n in range(max(1, n_lo), n_hi + 1):
        for ell, vis in [(0, 1.0), (1, 0.7), (2, 0.5)]:
            d02 = -0.15 * dnu if ell == 2 else 0.0
            nu_m = dnu * (n + ell / 2.0 + eps) + d02
            if nu_m < freq[0] or nu_m > freq[-1]:
                continue
            env = np.exp(-0.5 * ((nu_m - numax) / sigma_env) ** 2)
            modes += height * env * vis * width ** 2 / ((freq - nu_m) ** 2 + width ** 2)
    return modes

def forward_operator(freq, numax, delta_nu, sigma_env, mode_height, mode_width,
                     harvey_zeta1, harvey_nc1, harvey_zeta2, harvey_nc2, white_noise):
    """
    Forward model: Given oscillation parameters → power spectrum.
    
    Args:
        freq: frequency array
        numax: frequency of maximum oscillation power
        delta_nu: large frequency separation
        sigma_env: Gaussian envelope width
        mode_height: peak mode height
        mode_width: Lorentzian HWHM
        harvey_zeta1, harvey_nc1: Harvey component 1 parameters
        harvey_zeta2, harvey_nc2: Harvey component 2 parameters
        white_noise: white noise level
        
    Returns:
        y_pred: predicted power spectrum
    """
    # Background model
    background = bg_model(freq, harvey_zeta1, harvey_nc1, harvey_zeta2, harvey_nc2, white_noise)
    
    # Oscillation modes
    modes = osc_modes(freq, numax, delta_nu, sigma_env, mode_height, mode_width)
    
    # Total power spectrum
    y_pred = background + modes
    
    return y_pred
