import numpy as np


# --- Extracted Dependencies ---

def operation_xz(gsize):
    delta_xz = np.array([[[1, -1]], [[-1, 1]]], dtype='float32')
    xzfft = np.fft.fftn(delta_xz, gsize) * np.conj(np.fft.fftn(delta_xz, gsize))
    return xzfft
