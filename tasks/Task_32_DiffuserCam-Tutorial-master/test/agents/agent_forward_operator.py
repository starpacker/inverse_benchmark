import numpy as np

import numpy.fft as fft

def C(M_arr, full_size, sensor_size):
    # Crop operator
    top = (full_size[0] - sensor_size[0]) // 2
    left = (full_size[1] - sensor_size[1]) // 2
    return M_arr[top:top+sensor_size[0], left:left+sensor_size[1]]

def CT(b, full_size, sensor_size):
    # Transpose of Crop (Zero Pad)
    pad_top = (full_size[0] - sensor_size[0]) // 2
    pad_left = (full_size[1] - sensor_size[1]) // 2
    # Create full zero array and place b in center
    out = np.zeros(full_size, dtype=b.dtype)
    out[pad_top:pad_top+sensor_size[0], pad_left:pad_left+sensor_size[1]] = b
    return out

def M_func(vk, H_fft):
    # Convolution operator in freq domain
    # M(v) = Real(IFFT( FFT(v) * H ))
    # Note: Logic must match original: ifftshift before fft, fftshift after ifft
    return np.real(fft.fftshift(fft.ifft2(fft.fft2(fft.ifftshift(vk)) * H_fft)))

def precompute_H_fft(psf, full_size, sensor_size):
    # H = FFT( ZeroPad(PSF) )
    return fft.fft2(fft.ifftshift(CT(psf, full_size, sensor_size)))

def forward_operator(x, psf):
    """
    Simulates the forward imaging process: y = C(M(CT(x))) + Noise
    Inputs:
        x: Ground truth object (sensor_size)
        psf: Point Spread Function (sensor_size)
    Returns:
        y_pred: Simulated measurement
    """
    sensor_size = np.array(psf.shape)
    full_size = 2 * sensor_size
    
    # 1. Pad object to full domain (CT)
    x_padded = CT(x, full_size, sensor_size)
    
    # 2. Convolve (M)
    # We need H_fft for convolution
    H_fft = precompute_H_fft(psf, full_size, sensor_size)
    meas_full = M_func(x_padded, H_fft)
    
    # 3. Crop back to sensor size (C)
    y_clean = C(meas_full, full_size, sensor_size)
    
    return y_clean
