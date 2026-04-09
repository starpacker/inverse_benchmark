import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def normalize(v):
    norm = v.sum()
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v/norm

def gaussian_kernel(size, fwhmx=3, fwhmy=3, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return normalize(np.exp(-4 * np.log(2) * (((x - x0) ** 2) / fwhmx**2 + ((y - y0) ** 2) / fwhmy**2)))
