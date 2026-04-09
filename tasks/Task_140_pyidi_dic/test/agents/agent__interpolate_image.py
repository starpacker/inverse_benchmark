import matplotlib

matplotlib.use('Agg')

import os

from scipy.ndimage import map_coordinates

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def _interpolate_image(image, y_coords, x_coords):
    """Bilinear interpolation of image at fractional coordinates."""
    return map_coordinates(image, [y_coords, x_coords], order=1,
                           mode='reflect').reshape(y_coords.shape)
