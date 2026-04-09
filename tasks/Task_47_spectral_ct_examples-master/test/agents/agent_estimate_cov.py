import numpy as np

from scipy import signal

def estimate_cov(I1, I2):
    """
    Estimate the covariance of I1 and I2 using a Laplacian-like filter.
    This emphasizes high-frequency noise.
    """
    assert I1.shape == I2.shape
    H, W = I1.shape
    
    # Laplacian kernel to filter out smooth signal and keep noise
    M = np.array([[1, -2, 1],
                  [-2, 4., -2],
                  [1, -2, 1]])
    
    # Convolve and compute scalar product
    sigma = np.sum(signal.convolve2d(I1, M) * signal.convolve2d(I2, M))
    sigma /= (W * H - 1)
    
    # Normalization factor (empirical or derived from M)
    return sigma / 36.0
