import time
import numpy as np
from scipy import fft
from scipy.fftpack import next_fast_len

def rfft2_convolve_setup(psf, dtype=np.float32):
    """Set up FFT-based convolution parameters."""
    psf = psf.astype(dtype)
    psf_shape = np.array(psf.shape)
    padded_shape = 2 * psf_shape[-3:-1] - 1
    padded_shape = np.array([next_fast_len(int(i)) for i in padded_shape])
    padded_shape = list(np.r_[psf_shape[-4], padded_shape, psf.shape[-1]].astype(int))
    start_idx = ((np.array(padded_shape[-3:-1]) - psf_shape[-3:-1]) // 2).astype(int)
    end_idx = (start_idx + psf_shape[-3:-1]).astype(int)
    
    return {
        "psf": psf,
        "psf_shape": psf_shape,
        "padded_shape": padded_shape,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "dtype": dtype
    }

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
        shape = padded_shape

    vpad = np.zeros(shape).astype(v.dtype)
    vpad[..., int(start_idx[0]):int(end_idx[0]), int(start_idx[1]):int(end_idx[1]), :] = v
    return vpad

def crop_array(x, setup):
    """Crop array from padded_shape to original shape."""
    start_idx = setup["start_idx"]
    end_idx = setup["end_idx"]
    return x[..., int(start_idx[0]):int(end_idx[0]), int(start_idx[1]):int(end_idx[1]), :]

def compute_psf_fft(setup):
    """Compute FFT of padded PSF."""
    psf = setup["psf"]
    padded_psf = pad_array(psf, setup)
    H = fft.rfft2(padded_psf, axes=(-3, -2), norm="ortho")
    return H

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

def prox_nonneg(x):
    """Proximal operator for non-negativity constraint."""
    return np.maximum(x, 0)

def run_inversion(data_dict, n_iter=50):
    """
    Run APGD (FISTA) reconstruction algorithm.
    """
    psf = data_dict["psf"]
    measurement = data_dict["data"]
    dtype = psf.dtype
    
    setup = rfft2_convolve_setup(psf, dtype=dtype)
    H = compute_psf_fft(setup)
    H_adj = np.conj(H)
    
    print("Estimating Lipschitz constant...")
    L = power_method_lipschitz(setup, H, H_adj, max_iter=20)
    print(f"Lipschitz constant L = {L:.4e}")
    
    x_k = np.zeros_like(psf)
    y_k = x_k.copy()
    t_k = 1.0
    
    step_size = 1.0 / L if L > 0 else 1.0
    
    print(f"Starting APGD for {n_iter} iterations...")
    start_time = time.time()
    
    for i in range(n_iter):
        if i % 10 == 0:
            print(f"  Iteration {i}/{n_iter}")
        
        Ay_k = convolve_fft(y_k, H, setup, pad=True)
        residual = Ay_k - measurement
        gradient = deconvolve_fft(residual, H_adj, setup, pad=True)
        
        x_k_next_unprox = y_k - step_size * gradient
        
        x_k_next = prox_nonneg(x_k_next_unprox)
        
        t_k_next = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
        y_k = x_k_next + ((t_k - 1) / t_k_next) * (x_k_next - x_k)
        
        x_k = x_k_next
        t_k = t_k_next
    
    end_time = time.time()
    print(f"Reconstruction finished in {end_time - start_time:.2f}s")
    
    result = x_k
    if result.shape[0] == 1:
        result = result[0]
    
    return result