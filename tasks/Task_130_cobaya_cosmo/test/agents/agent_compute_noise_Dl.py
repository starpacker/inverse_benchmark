import warnings

import numpy as np

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use('Agg')

NOISE_UK_ARCMIN = 45.0

BEAM_ARCMIN = 7.0

def compute_noise_Dl(lmax):
    """Compute white-noise + Gaussian-beam noise D_l."""
    ell = np.arange(lmax + 1, dtype=float)
    nr = NOISE_UK_ARCMIN * np.pi / (180 * 60)
    sb = BEAM_ARCMIN * np.pi / (180 * 60) / np.sqrt(8 * np.log(2))
    Nl = nr**2 * np.exp(ell * (ell + 1) * sb**2)
    Dl = np.zeros_like(ell)
    Dl[2:] = ell[2:] * (ell[2:] + 1) / (2 * np.pi) * Nl[2:]
    return Dl
