import numpy as np

import matplotlib

matplotlib.use('Agg')

def compute_cc(a, b):
    """Compute cross-correlation."""
    a, b = a - np.mean(a), b - np.mean(b)
    d = np.sqrt(np.sum(a**2) * np.sum(b**2))
    return float(np.sum(a * b) / d) if d > 1e-30 else 0.0
