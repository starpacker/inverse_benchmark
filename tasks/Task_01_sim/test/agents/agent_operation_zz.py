import numpy as np


# --- Extracted Dependencies ---

def operation_zz(gsize):
    delta_zz = np.array([[[1]], [[-2]], [[1]]], dtype='float32')
    zzfft = np.fft.fftn(delta_zz, gsize) * np.conj(np.fft.fftn(delta_zz, gsize))
    return zzfft
