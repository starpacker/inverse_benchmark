import numpy as np
import numpy.fft as fft
def CT(b, full_size, sensor_size):
    pad_top = (full_size[0] - sensor_size[0]) // 2
    pad_left = (full_size[1] - sensor_size[1]) // 2
    out = np.zeros(full_size, dtype=b.dtype)
    out[pad_top:pad_top + sensor_size[0], pad_left:pad_left + sensor_size[1]] = b
    return out

def M_func(vk, H_fft):
    return np.real(fft.fftshift(fft.ifft2(fft.fft2(fft.ifftshift(vk)) * H_fft)))

def precompute_H_fft(psf, full_size, sensor_size):
    return fft.fft2(fft.ifftshift(CT(psf, full_size, sensor_size)))

def forward_operator(x, psf):
    
    
    sensor_size = x.shape
   
    full_size = (sensor_size[0] * 2, sensor_size[1] * 2)
    
    
    H_fft = precompute_H_fft(psf, full_size, sensor_size)
    
    x_padded = CT(x, full_size, sensor_size)
    
    y_clean = M_func(x_padded, H_fft)
    return y_clean