import numpy as np
from scipy import fft


# --- Extracted Dependencies ---

def pad_array(v, setup):
    """Pad array to padded_shape."""
    padded_shape = setup["padded_shape"]
    start_idx = setup["start_idx"]
    end_idx = setup["end_idx"]
    
    if len(v.shape) == 5:
        batch_size = v.shape[0]
        shape = [batch_size] + padded_shape
    elif len(v.shape) == 4:
        shape = padded_shape
    else:
        raise ValueError("Expected 4D or 5D tensor")
    
    vpad = np.zeros(shape).astype(v.dtype)
    vpad[..., int(start_idx[0]):int(end_idx[0]), int(start_idx[1]):int(end_idx[1]), :] = v
    return vpad

def crop_array(x, setup):
    """Crop array from padded_shape to original shape."""
    start_idx = setup["start_idx"]
    end_idx = setup["end_idx"]
    return x[..., int(start_idx[0]):int(end_idx[0]), int(start_idx[1]):int(end_idx[1]), :]

def deconvolve_fft(y, H_adj, setup, pad=True):
    """Perform adjoint convolution (correlation) using FFT."""
    padded_shape = setup["padded_shape"]
    
    if pad:
        y_padded = pad_array(y, setup)
    else:
        y_padded = y
    
    deconv_output = fft.rfft2(y_padded, axes=(-3, -2), norm="ortho") * H_adj
    deconv_output = fft.ifftshift(
        fft.irfft2(deconv_output, axes=(-3, -2), s=padded_shape[-3:-1], norm="ortho"),
        axes=(-3, -2),
    )
    
    if pad:
        deconv_output = crop_array(deconv_output, setup)
    
    return deconv_output.real.astype(setup["dtype"])
