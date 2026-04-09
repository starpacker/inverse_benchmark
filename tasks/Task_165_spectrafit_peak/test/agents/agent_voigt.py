import matplotlib

matplotlib.use('Agg')

import numpy as np

from scipy.special import voigt_profile

def voigt(x, amplitude, center, sigma, gamma):
    """Voigt peak – convolution of Gaussian and Lorentzian."""
    vp = voigt_profile(x - center, sigma, gamma)
    vp_max = voigt_profile(0.0, sigma, gamma)
    if vp_max > 0:
        return amplitude * vp / vp_max
    return np.zeros_like(x)
