import numpy as np

import matplotlib

matplotlib.use("Agg")

def build_convolution_matrix(wavelet, n):
    """
    Build the convolution matrix H such that H @ r ≈ fftconvolve(r, wavelet, 'same').
    """
    w_len = len(wavelet)
    half_w = w_len // 2
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(w_len):
            col = i - half_w + j
            if 0 <= col < n:
                H[i, col] = wavelet[j]
    return H
