import numpy as np

import matplotlib

matplotlib.use("Agg")

def compute_reflection_coefficients(eps_profile):
    """
    From a 1D permittivity profile ε(z), compute reflection coefficients.
    r(z) = (sqrt(ε(z+1)) - sqrt(ε(z))) / (sqrt(ε(z+1)) + sqrt(ε(z)))
    """
    sqrt_eps = np.sqrt(eps_profile)
    r = np.zeros_like(eps_profile)
    r[:-1] = (sqrt_eps[1:] - sqrt_eps[:-1]) / (sqrt_eps[1:] + sqrt_eps[:-1] + 1e-12)
    return r
