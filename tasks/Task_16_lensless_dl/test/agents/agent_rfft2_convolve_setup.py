import numpy as np
from scipy.fftpack import next_fast_len


# --- Extracted Dependencies ---

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
