import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def _image_gradients(image):
    """Compute image gradients using central differences."""
    gy = np.zeros_like(image)
    gx = np.zeros_like(image)
    gy[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2.0
    gx[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2.0
    return gy, gx
