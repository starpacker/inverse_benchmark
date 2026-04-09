import numpy as np

import matplotlib

matplotlib.use("Agg")

def _psnr(a, b):
    """Peak Signal-to-Noise Ratio."""
    mse = np.mean((a - b) ** 2)
    mx = np.max(np.abs(a))
    if mse < 1e-30:
        return 100.0
    if mx < 1e-30:
        return 0.0
    return float(10 * np.log10(mx ** 2 / mse))
