import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.ndimage import gaussian_filter

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_abundance_maps(img_size, n_end, rng):
    """
    Create spatially smooth, physically realistic abundance maps.
    """
    A = np.zeros((n_end, img_size, img_size))
    for i in range(n_end):
        raw = rng.standard_normal((img_size, img_size))
        A[i] = gaussian_filter(raw, sigma=8 + 2 * i)

    A_exp = np.exp(2.0 * (A - A.max(axis=0, keepdims=True)))
    A_sum = A_exp.sum(axis=0, keepdims=True)
    A_norm = A_exp / A_sum
    
    corners = [(0,0), (0,img_size-1), (img_size-1,0), (img_size-1,img_size-1)]
    for i in range(min(n_end, 4)):
        r, c = corners[i]
        A_norm[:, r, c] = 0.0
        A_norm[i, r, c] = 1.0

    return A_norm.reshape(n_end, -1)
