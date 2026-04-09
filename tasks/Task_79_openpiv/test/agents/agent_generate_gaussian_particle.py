import numpy as np

import matplotlib

matplotlib.use('Agg')

def generate_gaussian_particle(img, x0, y0, diameter, intensity=255.0):
    """Render a single Gaussian particle onto the image."""
    sigma = diameter / 4.0
    r = int(3 * sigma) + 1
    y_min = max(0, int(y0) - r)
    y_max = min(img.shape[0], int(y0) + r + 1)
    x_min = max(0, int(x0) - r)
    x_max = min(img.shape[1], int(x0) + r + 1)
    
    for iy in range(y_min, y_max):
        for ix in range(x_min, x_max):
            dist2 = (ix - x0)**2 + (iy - y0)**2
            img[iy, ix] += intensity * np.exp(-dist2 / (2 * sigma**2))
    return img
