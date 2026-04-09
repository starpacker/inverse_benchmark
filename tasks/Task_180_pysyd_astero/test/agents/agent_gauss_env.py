import numpy as np

import matplotlib

matplotlib.use('Agg')

def gauss_env(freq, amp, center, sigma):
    """Gaussian envelope function."""
    return amp * np.exp(-0.5 * ((freq - center) / sigma) ** 2)
