import numpy as np

import numpy.fft as fft

def CT(b, full_size, sensor_size):
    # Transpose of Crop (Zero Pad)
    pad_top = (full_size[0] - sensor_size[0]) // 2
    pad_left = (full_size[1] - sensor_size[1]) // 2
    # Create full zero array and place b in center
    out = np.zeros(full_size, dtype=b.dtype)
    out[pad_top:pad_top+sensor_size[0], pad_left:pad_left+sensor_size[1]] = b
    return out

def precompute_H_fft(psf, full_size, sensor_size):
    # H = FFT( ZeroPad(PSF) )
    return fft.fft2(fft.ifftshift(CT(psf, full_size, sensor_size)))
