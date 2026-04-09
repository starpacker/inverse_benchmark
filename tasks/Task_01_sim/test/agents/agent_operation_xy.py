import numpy as np


# --- Extracted Dependencies ---

def operation_xy(gsize):
    delta_xy = np.array([[[1, -1], [-1, 1]]], dtype='float32')
    xyfft = np.fft.fftn(delta_xy, gsize) * np.conj(np.fft.fftn(delta_xy, gsize))
    return xyfft
