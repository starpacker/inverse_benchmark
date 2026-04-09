import matplotlib

matplotlib.use('Agg')

import numpy as np

def gaussian(x, amplitude, center, sigma):
    """Gaussian peak."""
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
