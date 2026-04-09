import numpy as np

from scipy.optimize import curve_fit

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

def fit_gamma_variate(time_index, curve):
    try:
        # bounds: [t0, alpha, beta, amp]
        # constrained to be physically plausible
        popt, _ = curve_fit(
            gamma_variate, 
            time_index, 
            curve, 
            bounds=([0, 0.1, 0.1, 0], [20, 8, 8, np.max(curve)*2.5]),
            maxfev=2000
        )
        return popt
    except Exception:
        return None
