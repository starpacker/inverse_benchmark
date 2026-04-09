import math
import numpy as np


# --- Extracted Dependencies ---

def psnr(ref, img):
    """
    Peak signal-to-noise ratio (PSNR).
    """
    mse = np.mean((ref - img) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
