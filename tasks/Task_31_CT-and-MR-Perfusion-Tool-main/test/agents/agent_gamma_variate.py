import numpy as np

import warnings

warnings.filterwarnings("ignore")

def gamma_variate(t, t0, alpha, beta, amplitude=1.0):
    t = np.array(t)
    t_shifted = np.maximum(0, t - t0)
    result = np.zeros_like(t_shifted)
    mask = t > t0
    # Add small epsilon to avoid log(0) or div/0 issues implicitly
    safe_t = t_shifted[mask]
    result[mask] = amplitude * (safe_t**alpha) * np.exp(-safe_t/beta)
    return result
