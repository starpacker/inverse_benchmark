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

def load_and_preprocess_data(freq_min, freq_max, freq_res, seed,
                              harvey_zeta1, harvey_nc1, harvey_zeta2, harvey_nc2,
                              white_noise, gt_numax, gt_delta_nu, sigma_env,
                              mode_height, mode_width):
    """
    Synthesize asteroseismic power spectrum data.
    
    Returns:
        freq: frequency array (μHz)
        ps_true: true power spectrum
        ps_obs: observed (noisy) power spectrum
        params: dictionary of ground truth parameters
    """
    rng = np.random.RandomState(seed)
    freq = np.arange(freq_min, freq_max, freq_res)
    
    # Compute ground truth background
    bg_true = bg_model(freq, harvey_zeta1, harvey_nc1, harvey_zeta2, harvey_nc2, white_noise)
    
    # Compute oscillation modes
    modes = osc_modes(freq, gt_numax, gt_delta_nu, sigma_env, mode_height, mode_width)
    
    # True power spectrum
    ps_true = bg_true + modes
    
    # Observed power spectrum with exponential noise (chi-squared with 2 dof)
    ps_obs = ps_true * rng.exponential(1.0, size=len(freq))
    
    # Store parameters for later use
    params = {
        'gt_numax': gt_numax,
        'gt_delta_nu': gt_delta_nu,
        'sigma_env': sigma_env,
        'mode_height': mode_height,
        'mode_width': mode_width,
        'harvey_zeta1': harvey_zeta1,
        'harvey_nc1': harvey_nc1,
        'harvey_zeta2': harvey_zeta2,
        'harvey_nc2': harvey_nc2,
        'white_noise': white_noise
    }
    
    return freq, ps_true, ps_obs, params
