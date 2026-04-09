import numpy as np

import matplotlib

matplotlib.use('Agg')

def tsv_gradient(image):
    """Gradient of TSV(I) w.r.t. I."""
    ny, nx = image.shape
    grad = np.zeros_like(image)
    grad[:, :-1] -= 2 * (image[:, 1:] - image[:, :-1])
    grad[:, 1:] += 2 * (image[:, 1:] - image[:, :-1])
    grad[:-1, :] -= 2 * (image[1:, :] - image[:-1, :])
    grad[1:, :] += 2 * (image[1:, :] - image[:-1, :])
    return grad
