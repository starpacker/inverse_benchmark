import numpy as np


# --- Extracted Dependencies ---

def operation_yy(gsize):
    delta_yy = np.array([[[1], [-2], [1]]], dtype='float32')
    yyfft = np.fft.fftn(delta_yy, gsize) * np.conj(np.fft.fftn(delta_yy, gsize))
    return yyfft
