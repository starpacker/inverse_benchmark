import matplotlib

matplotlib.use('Agg')

import numpy as np

def soft_threshold(x, threshold):
    """Soft thresholding operator."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
