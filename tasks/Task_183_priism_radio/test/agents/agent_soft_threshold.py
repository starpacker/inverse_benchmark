import numpy as np

import matplotlib

matplotlib.use('Agg')

def soft_threshold(x, threshold):
    """Proximal operator for L1 norm."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)
