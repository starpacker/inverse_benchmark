import numpy as np

import matplotlib

matplotlib.use("Agg")

def _rmse(a, b):
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean((a - b) ** 2)))
