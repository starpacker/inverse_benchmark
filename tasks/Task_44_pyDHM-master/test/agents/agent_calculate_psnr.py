import numpy as np

from math import pi, sqrt, log10

def calculate_psnr(img1, img2):
    """Calculates PSNR between two normalized images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0 
    return 20 * log10(PIXEL_MAX / sqrt(mse))
