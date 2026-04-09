import numpy as np

import matplotlib

matplotlib.use("Agg")

def gaussian_2d(size, cx, cy, flux, fwhm):
    """2-D Gaussian centred at (cx, cy) with peak = `flux`."""
    sigma = fwhm / 2.355
    y, x = np.mgrid[:size, :size]
    return flux * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2))
