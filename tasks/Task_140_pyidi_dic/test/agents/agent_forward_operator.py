import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.ndimage import map_coordinates

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def forward_operator(ref_image, dx_field, dy_field):
    """
    Apply known spatially-varying displacement field to reference speckle image.
    
    Forward model: output(r, c) = input(r - dy[r,c], c - dx[r,c])
    
    Args:
        ref_image: Reference speckle image (height, width)
        dx_field: x-displacement field (height, width)
        dy_field: y-displacement field (height, width)
    
    Returns:
        warped: Deformed image (height, width)
    """
    h, w = ref_image.shape
    rr, cc = np.meshgrid(np.arange(h, dtype=np.float64),
                         np.arange(w, dtype=np.float64), indexing='ij')
    warped = map_coordinates(ref_image, [rr - dy_field, cc - dx_field],
                             order=3, mode='reflect')
    return warped
