import numpy as np

import matplotlib

matplotlib.use("Agg")

def find_peak_near(image, row, col, search_radius=10):
    """Find the peak pixel position near (row, col)."""
    ny, nx = image.shape
    r0 = max(0, int(row - search_radius))
    r1 = min(ny, int(row + search_radius + 1))
    c0 = max(0, int(col - search_radius))
    c1 = min(nx, int(col + search_radius + 1))
    sub = image[r0:r1, c0:c1]
    idx = np.unravel_index(np.argmax(sub), sub.shape)
    return r0 + idx[0], c0 + idx[1]
