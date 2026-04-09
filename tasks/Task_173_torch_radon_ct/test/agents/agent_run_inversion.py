import time

import warnings

warnings.filterwarnings('ignore')

import numpy as np

import matplotlib

matplotlib.use('Agg')

from skimage.transform import radon, iradon, resize

def run_inversion(sinogram, theta, output_size, method='FBP', filter_name='ramp', 
                  n_iter=200, init_fbp=False):
    """
    Run CT reconstruction inversion to recover image from sinogram.
    
    Implements multiple reconstruction methods:
    - FBP: Filtered Back Projection with various filters
    - SIRT: Simultaneous Iterative Reconstruction Technique
    
    Args:
        sinogram: Input sinogram (2D numpy array)
        theta: Array of projection angles in degrees
        output_size: Size of the output reconstructed image
        method: Reconstruction method ('FBP' or 'SIRT')
        filter_name: Filter for FBP ('ramp', 'shepp-logan', 'cosine', 'hamming')
        n_iter: Number of iterations for SIRT
        init_fbp: Whether to initialize SIRT with FBP result
        
    Returns:
        reconstruction: Reconstructed image (2D numpy array)
        elapsed_time: Computation time in seconds
    """
    t0 = time.time()
    
    if method == 'FBP':
        # Filtered Back Projection
        reconstruction = iradon(sinogram, theta=theta, output_size=output_size,
                                filter_name=filter_name, circle=True)
    
    elif method == 'SIRT':
        # SIRT: Simultaneous Iterative Reconstruction Technique
        # Landweber iteration: x_{k+1} = x_k + step * A^T(y - A*x_k)
        
        # Estimate operator norm via power iteration for step size
        def estimate_operator_norm(theta_vals, out_size, n_power_iter=15):
            x = np.random.RandomState(0).randn(out_size, out_size)
            norm_est = 1.0
            for _ in range(n_power_iter):
                Ax = radon(x, theta=theta_vals, circle=True)
                AtAx = iradon(Ax, theta=theta_vals, output_size=out_size, 
                             filter_name=None, circle=True)
                norm_est = np.sqrt(np.sum(AtAx**2) / (np.sum(x**2) + 1e-10))
                x = AtAx / (np.linalg.norm(AtAx) + 1e-10) * np.linalg.norm(x)
            return norm_est
        
        # Create circular mask
        def get_circle_mask(size):
            center = size // 2
            Y, X = np.ogrid[:size, :size]
            return ((X - center)**2 + (Y - center)**2) <= center**2
        
        norm_est = estimate_operator_norm(theta, output_size)
        step = 0.9 / (norm_est + 1e-6)
        circle_mask = get_circle_mask(output_size)
        
        # Initialize
        if init_fbp:
            x = iradon(sinogram, theta=theta, output_size=output_size,
                      filter_name='ramp', circle=True)
            x = np.maximum(x, 0) * circle_mask
        else:
            x = np.zeros((output_size, output_size))
        
        # Iterative refinement
        for i in range(n_iter):
            # Forward projection
            Ax = radon(x, theta=theta, circle=True)
            # Residual
            residual = sinogram - Ax
            # Back projection (adjoint)
            gradient = iradon(residual, theta=theta, output_size=output_size,
                             filter_name=None, circle=True)
            # Update with step
            x = x + step * gradient
            # Non-negativity constraint and circular mask
            x = np.maximum(x, 0) * circle_mask
        
        reconstruction = x
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    elapsed_time = time.time() - t0
    return reconstruction, elapsed_time
