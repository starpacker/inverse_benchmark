import numpy as np

import matplotlib

matplotlib.use('Agg')

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
