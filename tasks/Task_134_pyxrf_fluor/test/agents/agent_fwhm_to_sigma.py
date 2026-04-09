import numpy as np

import matplotlib

matplotlib.use('Agg')

def fwhm_to_sigma(fwhm):
    """Convert FWHM to Gaussian sigma."""
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
