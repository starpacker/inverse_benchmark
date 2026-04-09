import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

import healpy as hp

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def forward_operator(data):
    """
    Compute the naive pseudo-Cl from the masked sky map.
    This is the forward observation model: apply mask, compute anafast,
    and correct by f_sky. The result is biased by mode-coupling.

    Parameters
    ----------
    data : dict from load_and_preprocess_data

    Returns
    -------
    cl_pseudo : array, shape (lmax+1,) – f_sky-corrected pseudo-Cl
    """
    full_map = data['full_map']
    mask = data['mask']
    nside = data['nside']
    lmax = data['lmax']

    masked_map = full_map * mask
    cl_pseudo = hp.anafast(masked_map, lmax=lmax)
    f_sky = np.mean(mask ** 2)
    cl_pseudo_corrected = cl_pseudo / f_sky
    return cl_pseudo_corrected
