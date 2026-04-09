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
# Helper Classes (Inlined from lensless/recon/rfft_convolve.py and lensless/recon/admm.py)
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


class ADMM_Solver:
    """
    Object for applying ADMM with a non-negativity constraint and a total variation (TV) prior.
    Inlined from lensless.recon.admm
    """
    def __init__(self, psf, n_iter=50, mu1=1e-6, mu2=1e-5, mu3=4e-5, tau=0.0001, dtype=np.float32):
        self._psf = psf
        self._n_iter = n_iter
        self._dtype = dtype
        self._mu1 = mu1
        self._mu2 = mu2
        self._mu3 = mu3
        self._tau = tau
        
        # Initialize convolver (pad=False as ADMM handles padding strategy internally typically, 
        # but here we follow the simplified structure where we use padded shapes for optimization variables)
        # Actually, the original code sets pad=False for the convolver used inside ADMM, 
        # and works on padded variables directly.
        
        self._convolver = RealFFTConvolve2D(psf, dtype=dtype, pad=False)
        self._padded_shape = self._convolver._padded_shape
        self._psf_shape = self._convolver._psf_shape
        
        # Precompute TV Gram matrix
        self._PsiTPsi = self._finite_diff_gram(self._padded_shape, self._dtype)

    def _finite_diff(self, x):
        return np.stack(
            (np.roll(x, 1, axis=-3) - x, np.roll(x, 1, axis=-2) - x),
            axis=len(x.shape),
        )

    def _finite_diff_adj(self, x):
        diff1 = np.roll(x[..., 0], -1, axis=-3) - x[..., 0]
        diff2 = np.roll(x[..., 1], -1, axis=-2) - x[..., 1]
        return diff1 + diff2

    def _finite_diff_gram(self, shape, dtype):
        gram = np.zeros(shape, dtype=dtype)
        if shape[0] == 1:
            gram[0, 0, 0] = 4
            gram[0, 0, 1] = gram[0, 0, -1] = gram[0, 1, 0] = gram[0, -1, 0] = -1
        else:
            gram[0, 0, 0] = 6
            gram[0, 0, 1] = gram[0, 0, -1] = gram[0, 1, 0] = gram[0, -1, 0] = gram[1, 0, 0] = gram[-1, 0, 0] = -1
        return fft.rfft2(gram, axes=(-3, -2))

    def run(self, measurement):
        # Initialize variables
        # We work in the padded space
        image_est = np.zeros([1] + self._padded_shape, dtype=self._dtype)
        
        # Prepare data: pad it to match simulation size
        # The measurement is usually smaller or same size as PSF, but here we assume it matches PSF spatial dims
        # We need to embed the measurement into the padded field
        data_padded = self._convolver._pad(measurement)
        
        # Variables
        X = np.zeros_like(image_est)
        U = np.zeros_like(self._finite_diff(image_est))
        W = np.zeros_like(X)
        
        xi = np.zeros_like(image_est)
        eta = np.zeros_like(U)
        rho = np.zeros_like(X)
        
        # Precompute division matrices
        H = self._convolver._H
        Hadj = self._convolver._Hadj
        
        # R_divmat
        denom = self._mu1 * (np.abs(Hadj * H)) + self._mu2 * np.abs(self._PsiTPsi) + self._mu3
        R_divmat = 1.0 / denom.astype(np.complex64)
        
        # X_divmat
        X_divmat = 1.0 / (self._convolver._pad(np.ones(self._psf_shape, dtype=self._dtype)) + self._mu1)
        
        print(f"Starting ADMM for {self._n_iter} iterations...")
        
        for i in range(self._n_iter):
            if i % 10 == 0:
                print(f"  Iteration {i}/{self._n_iter}")
            
            # 1. U update (TV Soft Thresholding)
            Psi_out = self._finite_diff(image_est)
            U = np.sign(Psi_out + eta / self._mu2) * np.maximum(0, np.abs(Psi_out + eta / self._mu2) - self._tau / self._mu2)
            
            # 2. X update
            forward_out = self._convolver.convolve(image_est) # This convolve is unpadded because pad=False
            X = X_divmat * (xi + self._mu1 * forward_out + data_padded)
            
            # 3. W update (Non-negativity)
            W = np.maximum(rho / self._mu3 + image_est, 0)
            
            # 4. Image update (Frequency domain)
            rk = (
                (self._mu3 * W - rho)
                + self._finite_diff_adj(self._mu2 * U - eta)
                + self._convolver.deconvolve(self._mu1 * X - xi)
            )
            
            freq_result = R_divmat * fft.rfft2(rk, axes=(-3, -2))
            image_est = fft.irfft2(freq_result, axes=(-3, -2), s=self._convolver._padded_shape[-3:-1])
            
            # 5. Lagrangian updates
            forward_out = self._convolver.convolve(image_est)
            Psi_out = self._finite_diff(image_est)
            
            xi += self._mu1 * (forward_out - X)
            eta += self._mu2 * (Psi_out - U)
            rho += self._mu3 * (image_est - W)
            
        # Crop result
        return self._convolver._crop(image_est)

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
    convolver = RealFFTConvolve2D(psf, dtype=psf.dtype, pad=False) 
    return convolver.convolve(image_est)

# ==============================================================================
# 3. Run Inversion
# ==============================================================================

def run_inversion(data_dict, n_iter=50):
    psf = data_dict["psf"]
    measurement = data_dict["data"]
    
    solver = ADMM_Solver(psf, n_iter=n_iter)
    start_time = time.time()
    result = solver.run(measurement)
    end_time = time.time()
    
    print(f"Reconstruction finished in {end_time - start_time:.2f}s")
    
    # Result might have batch dim [1, H, W, C]
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
    ax.set_title("ADMM Reconstruction (Inlined)")
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
    evaluate_results(reconstruction, "admm_result_inlined.png")
    
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")
```