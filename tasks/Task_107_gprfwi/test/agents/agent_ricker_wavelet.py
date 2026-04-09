import numpy as np

import matplotlib

matplotlib.use("Agg")

def ricker_wavelet(points, a):
    """Ricker wavelet (Mexican hat). Equivalent to scipy.signal.ricker."""
    A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
    wsq = a**2
    vec = np.arange(0, points) - (points - 1.0) / 2
    tsq = vec**2
    mod = (1 - tsq / wsq)
    gauss = np.exp(-tsq / (2 * wsq))
    return A * mod * gauss
