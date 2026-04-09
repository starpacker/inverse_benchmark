import numpy as np

import matplotlib

matplotlib.use("Agg")

def aperture_sum(image, row, col, radius):
    """Sum of pixel values inside a circular aperture."""
    ny, nx = image.shape
    yy, xx = np.mgrid[:ny, :nx]
    mask = (yy - row) ** 2 + (xx - col) ** 2 <= radius ** 2
    return image[mask].sum(), mask
