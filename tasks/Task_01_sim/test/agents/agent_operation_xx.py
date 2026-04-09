import numpy as np


# --- Extracted Dependencies ---

def operation_xx(gsize):
    delta_xx = np.array([[[1, -2, 1]]], dtype='float32')
    xxfft = np.fft.fftn(delta_xx, gsize) * np.conj(np.fft.fftn(delta_xx, gsize))
    return xxfft
