```python
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from scipy.fftpack import next_fast_len
from lensless.utils.io import load_data
from lensless.utils.plot import plot_image
import argparse

# ==============================================================================
# Helper Classes (Inlined from lensless/recon/rfft_convolve.py and lensless/recon/apgd.py)
# ==============================================================================

class RealFFTConvolve2D:
    """
    2D convolution in Fourier domain, with same real-valued kernel.
    Inlined from lensless.recon.rfft_convolve
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

        # cropping / padding indexes
        self._padded_shape = 2 * self._psf_shape[-3:-1] - 1
        self._padded_shape = np.array([next_fast_len(i) for i in self._padded_shape])
        self._padded_shape = list(np.r_[self._psf_shape[-4], self._padded_shape, self._psf.shape[-1]])
        
        self._start_idx = (self._padded_shape[-3:-1] - self._psf_shape[-3:-1]) // 2
        self._end_idx = self._start_idx + self._psf_shape[-3:-1]

        # precompute filter in frequency domain
        self._H = fft.rfft2(self._pad(self._psf), axes=(-3, -2), norm=self.norm)
        self._Hadj = np.conj(self._H)
        self._padded_data = np.zeros(self._padded_shape).astype(self.dtype)

    def convolve(self, x):
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


class APGD_Solver:
    """
    Object for applying Accelerated Proximal Gradient Descent (APGD / FISTA).
    Simplified from lensless.recon.apgd
    """
    def __init__(self, psf, n_iter=50, dtype=np.float32, lambda_reg=0.001):
        self._psf = psf
        self._n_iter = n_iter
        self._dtype = dtype
        self._lambda = lambda_reg
        
        # Convolver (pad=False, we handle optimization in padded space usually, 
        # but to keep simple we use the same strategy as ADMM or direct convolution)
        # For FISTA, usually we compute gradient in image space.
        # Let's use the convolver with padding=True to simplify the gradient calculation in valid region
        # OR better: use same strategy as ADMM (pad=False) and work in padded domain for efficiency.
        # Let's stick to valid convolution (pad=True) which is more intuitive but slower, 
        # or use the efficient one. The library APGD uses pad=False internally implicitly via pycsou operators.
        
        # We will implement FISTA on the unpadded image directly using the convolve/deconvolve operators
        self._convolver = RealFFTConvolve2D(psf, dtype=dtype, pad=True)
        
        # Lipschitz constant estimation (Power Method)
        print("Estimating Lipschitz constant...")
        self._L = self._power_method(max_iter=20)
        print(f"Lipschitz constant L = {self._L:.4e}")

    def _power_method(self, max_iter=20):
        # Power iteration to find largest eigenvalue of H'H
        x = np.random.randn(*self._psf.shape).astype(self._dtype)
        x /= np.linalg.norm(x)
        
        for _ in range(max_iter):
            x = self._convolver.deconvolve(self._convolver.convolve(x))
            norm = np.linalg.norm(x)
            x /= norm
            
        return norm

    def _prox_nonneg(self, x):
        return np.maximum(x, 0)
        
    def _prox_l1(self, x, thresh):
        return np.sign(x) * np.maximum(0, np.abs(x) - thresh)

    def run(self, measurement):
        # FISTA / APGD
        # Minimize 0.5 * ||Ax - y||^2 + lambda * R(x)
        # Gradient of data fidelity: A'(Ax - y)
        
        x_k = np.zeros_like(self._psf) # Initial estimate
        y_k = x_k.copy()
        t_k = 1.0
        
        step_size = 1.0 / self._L
        
        print(f"Starting APGD for {self._n_iter} iterations...")
        
        for i in range(self._n_iter):
            if i % 10 == 0:
                print(f"  Iteration {i}/{self._n_iter}")
                
            # Gradient step on y_k
            # grad = A'(A(y_k) - b)
            residual = self._convolver.convolve(y_k) - measurement
            gradient = self._convolver.deconvolve(residual)
            
            x_k_next_unprox = y_k - step_size * gradient
            
            # Proximal step (Non-negativity)
            x_k_next = self._prox_nonneg(x_k_next_unprox)
            
            # Momentum update (FISTA)
            t_k_next = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
            y_k = x_k_next + ((t_k - 1) / t_k_next) * (x_k_next - x_k)
            
            x_k = x_k_next
            t_k = t_k_next
            
        return x_k

# ==============================================================================
# 1. Load and Preprocess Data
# ==============================================================================

def load_and_preprocess_data(psf_path, data_path, downsample=4):
    print(f"Loading data from {data_path}...")
    print(f"Loading PSF from {psf_path}...")
    
    psf, data = load_data(
        psf_fp=psf_path,
        data_fp=data_path,
        background_fp=None,
        dtype="float32",
        downsample=downsample,
        bayer=False,
        plot=False,
        flip=False,
        normalize=True
    )
    
    print(f"Data shape: {data.shape}")
    print(f"PSF shape: {psf.shape}")
    
    return {"psf": psf, "data": data}

# ==============================================================================
# 2. Forward Operator
# ==============================================================================

def forward_operator(image_est, psf):
    convolver = RealFFTConvolve2D(psf, dtype=psf.dtype, pad=True) 
    return convolver.convolve(image_est)

# ==============================================================================
# 3. Run Inversion
# ==============================================================================

def run_inversion(data_dict, n_iter=50):
    psf = data_dict["psf"]
    measurement = data_dict["data"]
    
    solver = APGD_Solver(psf, n_iter=n_iter)
    start_time = time.time()
    result = solver.run(measurement)
    end_time = time.time()
    
    print(f"Reconstruction finished in {end_time - start_time:.2f}s")
    
    if result.shape[0] == 1:
        result = result[0]
        
    return result

# ==============================================================================
# 4. Evaluate Results
# ==============================================================================

def evaluate_results(reconstruction, output_path="result.png"):
    print(f"Saving result to {output_path}...")
    ax = plot_image(reconstruction, gamma=None)
    if hasattr(ax, "__len__"):
        ax = ax[0,0]
    ax.set_title("APGD Reconstruction (Inlined)")
    plt.savefig(output_path)
    plt.close()
    
    npy_path = output_path.replace(".png", ".npy")
    np.save(npy_path, reconstruction)
    print(f"Saved numpy array to {npy_path}")

# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    psf_file = "data/psf/tape_rgb.png"
    data_file = "data/raw_data/thumbs_up_rgb.png"
    
    if not os.path.exists(psf_file):
        print(f"Error: {psf_file} not found.")
        exit(1)
        
    data_dict = load_and_preprocess_data(psf_file, data_file, downsample=4)
    reconstruction = run_inversion(data_dict, n_iter=5)
    evaluate_results(reconstruction, "apgd_result_inlined.png")
    
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")
```