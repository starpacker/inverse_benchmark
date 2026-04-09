import time  # CRITICAL: Required for performance timing inside run_inversion
import numpy as np
from scipy import fft
from scipy.fftpack import next_fast_len
import matplotlib.pyplot as plt

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
        # Crops the central region of the padded array to match original dimensions
        return x[..., self._start_idx[0] : self._end_idx[0], self._start_idx[1] : self._end_idx[1], :]

    def _pad(self, v):
        # Pads the input array to the size required for linear convolution via FFT
        if len(v.shape) == 5:
            batch_size = v.shape[0]
            shape = [batch_size] + self._padded_shape
        elif len(v.shape) == 4:
            shape = self._padded_shape
        else:
            raise ValueError("Expected 4D or 5D tensor")

        vpad = np.zeros(shape).astype(v.dtype)
        # Insert v into the center of the zero-filled array
        vpad[..., self._start_idx[0] : self._end_idx[0], self._start_idx[1] : self._end_idx[1], :] = v
        return vpad

    def set_psf(self, psf):
        self._psf = psf.astype(self.dtype)
        self._psf_shape = np.array(self._psf.shape)

        # Calculate optimal shape for FFT (Power of 2 or composite of small primes)
        self._padded_shape = 2 * self._psf_shape[-3:-1] - 1
        self._padded_shape = np.array([next_fast_len(int(i)) for i in self._padded_shape])
        self._padded_shape = list(np.r_[self._psf_shape[-4], self._padded_shape, self._psf.shape[-1]].astype(int))
        
        # Calculate indices for padding/cropping
        self._start_idx = ((np.array(self._padded_shape[-3:-1]) - self._psf_shape[-3:-1]) // 2).astype(int)
        self._end_idx = (self._start_idx + self._psf_shape[-3:-1]).astype(int)

        # Pre-compute Optical Transfer Function (OTF)
        self._H = fft.rfft2(self._pad(self._psf), axes=(-3, -2), norm=self.norm)
        self._Hadj = np.conj(self._H)
        self._padded_data = np.zeros(self._padded_shape).astype(self.dtype)

    def convolve(self, x):
        # Forward operator: H * x
        if self.pad:
            self._padded_data = self._pad(x)
        else:
            self._padded_data[:] = x

        conv_output = fft.rfft2(self._padded_data, axes=(-3, -2)) * self._H
        conv_output = fft.ifftshift(
            fft.irfft2(conv_output, axes=(-3, -2), s=self._padded_shape[-3:-1]),
            axes=(-3, -2),
        )
        
        if self.pad:
            conv_output = self._crop(conv_output)
            
        return conv_output

    def deconvolve(self, y):
        # Adjoint operator: H^T * y (Correlation)
        if self.pad:
            self._padded_data = self._pad(y)
        else:
            self._padded_data[:] = y

        deconv_output = fft.rfft2(self._padded_data, axes=(-3, -2)) * self._Hadj
        deconv_output = fft.ifftshift(
            fft.irfft2(deconv_output, axes=(-3, -2), s=self._padded_shape[-3:-1]),
            axes=(-3, -2),
        )

        if self.pad:
            deconv_output = self._crop(deconv_output)
            
        return deconv_output

def run_inversion(data_dict, n_iter=50, mu1=1e-6, mu2=1e-5, mu3=4e-5, tau=0.0001):
    """
    Run ADMM-based inversion to reconstruct the image from measurements.
    """
    psf = data_dict["psf"]
    measurement = data_dict["data"]
    dtype = np.float32
    
    # Initialize convolver with pad=False for internal ADMM operations
    # We handle padding manually in the loop to keep dimensions consistent in Fourier domain
    convolver = RealFFTConvolve2D(psf, dtype=dtype, pad=False)
    padded_shape = convolver._padded_shape
    psf_shape = convolver._psf_shape
    
    # --- Helper functions for finite differences (TV prior) ---
    def finite_diff(x):
        # Computes gradient (dx, dy)
        return np.stack(
            (np.roll(x, 1, axis=-3) - x, np.roll(x, 1, axis=-2) - x),
            axis=len(x.shape),
        )
    
    def finite_diff_adj(x):
        # Computes divergence (transpose of gradient)
        diff1 = np.roll(x[..., 0], -1, axis=-3) - x[..., 0]
        diff2 = np.roll(x[..., 1], -1, axis=-2) - x[..., 1]
        return diff1 + diff2
    
    def finite_diff_gram(shape, dt):
        # Computes Fourier transform of the Laplacian kernel
        gram = np.zeros(shape, dtype=dt)
        if shape[0] == 1:
            gram[0, 0, 0] = 4
            gram[0, 0, 1] = gram[0, 0, -1] = gram[0, 1, 0] = gram[0, -1, 0] = -1
        else:
            gram[0, 0, 0] = 6
            gram[0, 0, 1] = gram[0, 0, -1] = gram[0, 1, 0] = gram[0, -1, 0] = gram[1, 0, 0] = gram[-1, 0, 0] = -1
        return fft.rfft2(gram, axes=(-3, -2))
    
    # Precompute TV Gram matrix
    PsiTPsi = finite_diff_gram(padded_shape, dtype)
    
    # Initialize variables in padded space
    image_est = np.zeros([1] + padded_shape, dtype=dtype)
    
    # Prepare data: pad measurement to match padded field
    data_padded = convolver._pad(measurement)
    
    # ADMM variables
    X = np.zeros_like(image_est)
    U = np.zeros_like(finite_diff(image_est))
    W = np.zeros_like(X)
    
    xi = np.zeros_like(image_est)
    eta = np.zeros_like(U)
    rho = np.zeros_like(X)
    
    # Precompute division matrices (The "Inversion" part)
    H = convolver._H
    Hadj = convolver._Hadj
    
    # Denominator for Tikhonov regularization in Fourier domain
    denom = mu1 * (np.abs(Hadj * H)) + mu2 * np.abs(PsiTPsi) + mu3
    R_divmat = 1.0 / denom.astype(np.complex64)
    
    X_divmat = 1.0 / (convolver._pad(np.ones(list(psf_shape.astype(int)), dtype=dtype)) + mu1)
    
    print(f"Starting ADMM for {n_iter} iterations...")
    start_time = time.time()
    
    for i in range(n_iter):
        if i % 10 == 0:
            print(f"  Iteration {i}/{n_iter}")
        
        # 1. U update (TV Soft Thresholding)
        Psi_out = finite_diff(image_est)
        U = np.sign(Psi_out + eta / mu2) * np.maximum(0, np.abs(Psi_out + eta / mu2) - tau / mu2)
        
        # 2. X update
        forward_out = convolver.convolve(image_est)
        X = X_divmat * (xi + mu1 * forward_out + data_padded)
        
        # 3. W update (Non-negativity)
        W = np.maximum(rho / mu3 + image_est, 0)
        
        # 4. Image update (Frequency domain inversion)
        rk = (
            (mu3 * W - rho)
            + finite_diff_adj(mu2 * U - eta)
            + convolver.deconvolve(mu1 * X - xi)
        )
        
        freq_result = R_divmat * fft.rfft2(rk, axes=(-3, -2))
        image_est = fft.irfft2(freq_result, axes=(-3, -2), s=convolver._padded_shape[-3:-1])
        
        # 5. Lagrangian updates
        forward_out = convolver.convolve(image_est)
        Psi_out = finite_diff(image_est)
        
        xi += mu1 * (forward_out - X)
        eta += mu2 * (Psi_out - U)
        rho += mu3 * (image_est - W)
    
    end_time = time.time()
    print(f"Reconstruction finished in {end_time - start_time:.2f}s")
    
    # Crop result to original size
    result = convolver._crop(image_est)
    
    # Remove batch dimension if present
    if result.shape[0] == 1:
        result = result[0]
    
    return result