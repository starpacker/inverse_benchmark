import numpy as np

import numpy.fft as fft

def M_func(vk, H_fft):
    # Convolution operator in freq domain
    # M(v) = Real(IFFT( FFT(v) * H ))
    # Note: Logic must match original: ifftshift before fft, fftshift after ifft
    return np.real(fft.fftshift(fft.ifft2(fft.fft2(fft.ifftshift(vk)) * H_fft)))
