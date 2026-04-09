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

def convolve_fft(x, H, setup, pad=True):
    """Perform convolution using FFT."""
    padded_shape = setup["padded_shape"]
    
    if pad:
        x_padded = pad_array(x, setup)
    else:
        x_padded = x
    
    conv_output = fft.rfft2(x_padded, axes=(-3, -2), norm="ortho") * H
    conv_output = fft.ifftshift(
        fft.irfft2(conv_output, axes=(-3, -2), s=padded_shape[-3:-1], norm="ortho"),
        axes=(-3, -2),
    )
    
    if pad:
        conv_output = crop_array(conv_output, setup)
    
    return conv_output.real.astype(setup["dtype"])

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

def power_method_lipschitz(setup, H, H_adj, max_iter=20):
    """Estimate Lipschitz constant using power method."""
    psf_shape = setup["psf_shape"]
    dtype = setup["dtype"]
    
    x = np.random.randn(*psf_shape).astype(dtype)
    x /= np.linalg.norm(x)
    
    for _ in range(max_iter):
        conv_x = convolve_fft(x, H, setup, pad=True)
        x = deconvolve_fft(conv_x, H_adj, setup, pad=True)
        norm_val = np.linalg.norm(x)
        if norm_val > 0:
            x /= norm_val
    
    return norm_val
