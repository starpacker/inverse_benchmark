import numpy as np

import matplotlib

matplotlib.use("Agg")

def source_time_function(t, t0, half_width=0.2):
    """Gaussian source-time function centred at t0."""
    return np.exp(-((t - t0) ** 2) / (2.0 * half_width ** 2))
