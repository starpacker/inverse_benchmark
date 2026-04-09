import numpy as np

import scipy.fft

def radon_transform_logic(image, theta):
    """
    Explicit implementation of the Radon Transform (Forward Projector).
    """
    num_angles = len(theta)
    N = image.shape[1] 
    sinogram = np.zeros((num_angles, N), dtype=np.float32)

    for i, angle in enumerate(theta):
        # Rotate the image. order=1 (linear)
        rotated = scipy.ndimage.rotate(image, -angle, reshape=False, order=1, mode='constant', cval=0.0)
        sinogram[i] = rotated.sum(axis=0)
        
    return sinogram

def filter_sinogram(sinogram, window=None):
    """
    Applies the Ram-Lak filter with an optional window function.
    """
    num_angles, num_detectors = sinogram.shape
    
    # Pad to the next power of 2 for efficient FFT
    n = num_detectors
    padded_len = max(64, int(2 ** np.ceil(np.log2(2 * n))))
    
    # Compute frequency axis
    freq = scipy.fft.rfftfreq(padded_len)
    
    # Ram-Lak filter: |f| (ramp filter)
    filt = 2 * np.abs(freq) 
    
    # Apply window
    if window == 'hann':
        w = np.hanning(2 * len(freq))[:len(freq)]
        filt *= w
    elif window == 'hamming':
        w = np.hamming(2 * len(freq))[:len(freq)]
        filt *= w
    
    # Apply filter in Fourier domain
    sino_fft = scipy.fft.rfft(sinogram, n=padded_len, axis=1)
    filtered_sino_fft = sino_fft * filt
    filtered_sino = scipy.fft.irfft(filtered_sino_fft, n=padded_len, axis=1)
    
    # Crop back to original size
    return filtered_sino[:, :num_detectors]

def backproject(sinogram, theta):
    """
    Explicit Backprojection algorithm.
    """
    num_angles, num_detectors = sinogram.shape
    N = num_detectors
    recon = np.zeros((N, N), dtype=np.float32)
    
    for i, angle in enumerate(theta):
        projection = sinogram[i]
        tiled_projection = np.tile(projection, (N, 1))
        
        # Rotate it back to the original angle
        rotated = scipy.ndimage.rotate(tiled_projection, -angle, reshape=False, order=1, mode='constant', cval=0.0)
        recon += rotated
        
    # Scale factor approximation
    return recon * (np.pi / (2 * num_angles))

def fbp_reconstruct(sinogram, theta, window=None):
    """
    Complete FBP pipeline: Filter -> Backproject.
    """
    filtered = filter_sinogram(sinogram, window=window)
    recon = backproject(filtered, theta)
    return recon

def sirt_reconstruct(sinogram, theta, n_iter=10):
    """
    Simultaneous Iterative Reconstruction Technique (SIRT).
    """
    num_angles, num_detectors = sinogram.shape
    N = num_detectors
    
    recon = np.zeros((N, N), dtype=np.float32)
    
    # Calculate Row Sums (R)
    ones_img = np.ones((N, N), dtype=np.float32)
    row_sums = radon_transform_logic(ones_img, theta)
    row_sums[row_sums == 0] = 1.0
    
    # Calculate Column Sums (C)
    ones_sino = np.ones_like(sinogram)
    col_sums = backproject(ones_sino, theta)
    col_sums[col_sums == 0] = 1.0
    
    for k in range(n_iter):
        fp = radon_transform_logic(recon, theta)
        diff = sinogram - fp
        correction_term = diff / row_sums
        correction = backproject(correction_term, theta)
        
        recon += correction / col_sums
        recon[recon < 0] = 0
        
    return recon

def run_inversion(sinogram, theta, method='fbp', **kwargs):
    """
    Runs the specified reconstruction algorithm.
    """
    if method == 'fbp':
        window = kwargs.get('window', None)
        return fbp_reconstruct(sinogram, theta, window=window)
    elif method == 'sirt':
        n_iter = kwargs.get('n_iter', 20)
        return sirt_reconstruct(sinogram, theta, n_iter=n_iter)
    else:
        raise ValueError(f"Unknown inversion method: {method}")
