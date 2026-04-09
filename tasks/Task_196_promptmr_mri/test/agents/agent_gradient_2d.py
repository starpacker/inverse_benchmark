import numpy as np

import matplotlib

matplotlib.use('Agg')

def gradient_2d(img):
    """Compute discrete gradient (finite differences)."""
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, :-1] = img[:, 1:] - img[:, :-1]
    gy[:-1, :] = img[1:, :] - img[:-1, :]
    return gx, gy
