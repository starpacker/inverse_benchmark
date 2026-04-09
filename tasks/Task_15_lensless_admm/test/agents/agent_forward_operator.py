import numpy as np
from scipy import fft
from scipy.fftpack import next_fast_len

class RealFFTConvolve2D:
    """
    2D convolution in Fourier domain, with same real-valued kernel.
    """
    def __init__(self, psf, dtype=np.float32, pad=True, norm="ortho"):
        self.dtype = dtype
        self.pad = pad
        self.norm = norm
        self.set_psf(psf)

    def _crop(self, x):
        return x[..., self._start_idx[0] : self._end_idx[0], self._start_idx[1] : self._end_idx[1], :]

    def _pad(self, v):
        if len(v.shape) == 5:
            batch_size = v.shape[0]
            shape = [batch_size] + self._padded_shape
        elif len(v.shape) == 4:
            shape = self._padded_shape
        else:
            raise ValueError("Expected 4D or 5D tensor")

        vpad = np.zeros(shape).astype(v.dtype)
        vpad[..., self._start_idx[0] : self._end_idx[0], self._start_idx[1] : self._end_idx[1], :] = v
        return vpad

    def set_psf(self, psf):
        self._psf = psf.astype(self.dtype)
        self._psf_shape = np.array(self._psf.shape)

        # Calculate padded shape: roughly 2x the original size
        self._padded_shape = 2 * self._psf_shape[-3:-1] - 1
        # Optimize shape for FFT speed
        self._padded_shape = np.array([next_fast_len(int(i)) for i in self._padded_shape])
        # Reconstruct full shape list (preserving batch/channel dims if present)
        self._padded_shape = list(np.r_[self._psf_shape[-4], self._padded_shape, self._psf.shape[-1]].astype(int))
        
        # Calculate centering indices
        self._start_idx = ((np.array(self._padded_shape[-3:-1]) - self._psf_shape[-3:-1]) // 2).astype(int)
        self._end_idx = (self._start_idx + self._psf_shape[-3:-1]).astype(int)

        # Pre-compute Transfer Function
        self._H = fft.rfft2(self._pad(self._psf), axes=(-3, -2), norm=self.norm)
        self._Hadj = np.conj(self._H)
        self._padded_data = np.zeros(self._padded_shape).astype(self.dtype)

    def convolve(self, x):
        if self.pad:
            self._padded_data = self._pad(x)
        else:
            self._padded_data[:] = x

        # Frequency domain multiplication
        conv_output = fft.rfft2(self._padded_data, axes=(-3, -2)) * self._H
        
        # Inverse transform and shift
        conv_output = fft.ifftshift(
            fft.irfft2(conv_output, axes=(-3, -2), s=self._padded_shape[-3:-1]),
            axes=(-3, -2),
        )
        
        if self.pad:
            conv_output = self._crop(conv_output)
            
        return conv_output

    def deconvolve(self, y):
        if self.pad:
            self._padded_data = self._pad(y)
        else:
            self._padded_data[:] = y

        # Multiply by conjugate (Adjoint operation)
        deconv_output = fft.rfft2(self._padded_data, axes=(-3, -2)) * self._Hadj
        
        deconv_output = fft.ifftshift(
            fft.irfft2(deconv_output, axes=(-3, -2), s=self._padded_shape[-3:-1]),
            axes=(-3, -2),
        )

        if self.pad:
            deconv_output = self._crop(deconv_output)
            
        return deconv_output

def forward_operator(image_est, psf):
    """
    Apply the forward imaging model: convolve the estimated image with the PSF.
    
    Parameters
    ----------
    image_est : np.ndarray
        Estimated image, shape [D, H, W, C] or [1, D, H, W, C].
    psf : np.ndarray
        Point spread function, shape [D, H, W, C].
    
    Returns
    -------
    np.ndarray
        Simulated measurement (convolution result).
    """
    convolver = RealFFTConvolve2D(psf, dtype=psf.dtype, pad=True)
    return convolver.convolve(image_est)