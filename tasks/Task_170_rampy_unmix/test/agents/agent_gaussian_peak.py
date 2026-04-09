import matplotlib

matplotlib.use('Agg')

import numpy as np

import warnings

warnings.filterwarnings('ignore', category=UserWarning)

def gaussian_peak(x, center, width, height):
    """Single Gaussian peak."""
    return height * np.exp(-0.5 * ((x - center) / width) ** 2)
