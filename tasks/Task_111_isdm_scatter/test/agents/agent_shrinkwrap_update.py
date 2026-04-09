import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, binary_fill_holes

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR  = "/data/yjh/website_assets/Task_111_isdm_scatter"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def shrinkwrap_update(g, support, sigma=2.0, threshold_fraction=0.1):
    """
    Shrinkwrap support update: blur the current estimate and threshold.
    """
    blurred = gaussian_filter(np.abs(g), sigma=sigma)
    threshold = threshold_fraction * np.max(blurred)
    new_support = blurred > threshold
    return new_support
