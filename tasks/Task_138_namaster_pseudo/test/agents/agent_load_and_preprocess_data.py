import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

import healpy as hp

import pymaster as nmt

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_and_preprocess_data(nside=64, lmax=None):
    """
    Generate simulated CMB map with known Cl and a galactic mask.
    Returns all data needed for subsequent steps.

    Returns
    -------
    data : dict with keys:
        'cl_true'  : array, shape (lmax+1,) – input theoretical power spectrum
        'full_map' : array, shape (npix,) – full-sky realization
        'mask'     : array, shape (npix,) – apodized binary mask
        'nside'    : int
        'lmax'     : int
    """
    if lmax is None:
        lmax = 2 * nside

    # Theoretical power spectrum (CMB-like)
    ell = np.arange(lmax + 1, dtype=float)
    cl_true = np.zeros(lmax + 1)
    cl_true[2:] = 1.0 / (ell[2:] * (ell[2:] + 1))
    cl_true *= 1e4  # scale for realistic amplitude

    # Full-sky realization
    np.random.seed(42)
    full_map = hp.synfast(cl_true, nside, lmax=lmax, verbose=False)

    # Galactic mask: cut |b| < 20°
    npix = hp.nside2npix(nside)
    mask = np.ones(npix)
    theta, _ = hp.pix2ang(nside, np.arange(npix))
    lat = np.pi / 2 - theta
    mask[np.abs(lat) < np.radians(20)] = 0

    # Apodize mask (C1 taper, 10° scale)
    mask_apo = nmt.mask_apodization(mask, 10.0, apotype='C1')

    data = {
        'cl_true': cl_true,
        'full_map': full_map,
        'mask': mask_apo,
        'nside': nside,
        'lmax': lmax,
    }
    return data
