import numpy as np


# --- Extracted Dependencies ---

def prepare_4d_array(img):
    """Ensure image is 4D: (depth, height, width, channels)."""
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    if len(img.shape) == 3:
        img = img[np.newaxis, :, :, :]
    return img
