import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

import sys

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")

if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

def create_gaussian_kernel(sigma, size=None):
    """Create a 2D Gaussian convolution kernel."""
    if size is None:
        size = int(6 * sigma + 1)
        if size % 2 == 0:
            size += 1
    half = size // 2
    y, x = np.mgrid[-half:half+1, -half:half+1].astype(np.float64)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel
