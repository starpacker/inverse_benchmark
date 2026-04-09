import numpy as np

import numpy.fft as fft

def MT_func(x, H_fft):
    # Adjoint of convolution
    x_zeroed = fft.ifftshift(x)
    return np.real(fft.fftshift(fft.ifft2(fft.fft2(x_zeroed) * np.conj(H_fft))))
