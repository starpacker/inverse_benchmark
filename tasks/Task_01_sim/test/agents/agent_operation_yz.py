import numpy as np


# --- Extracted Dependencies ---

def operation_yz(gsize):
    delta_yz = np.array([[[1], [-1]], [[-1], [1]]], dtype='float32')
    yzfft = np.fft.fftn(delta_yz, gsize) * np.conj(np.fft.fftn(delta_yz, gsize))
    return yzfft
