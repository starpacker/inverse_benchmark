import numpy as np

import matplotlib

matplotlib.use('Agg')

from skimage.data import shepp_logan_phantom as slp

from skimage.transform import resize, radon, iradon

def load_and_preprocess_data(image_size, n_angles_full, n_angles_sparse, noise_level):
    """
    Generate phantom and create sparse-view sinogram with noise.
    
    Parameters:
    -----------
    image_size : int
        Size of the output phantom image (image_size x image_size)
    n_angles_full : int
        Number of angles for full-view reference
    n_angles_sparse : int
        Number of angles for sparse-view acquisition
    noise_level : float
        Noise level as fraction of max sinogram value
        
    Returns:
    --------
    phantom : ndarray
        Ground truth phantom image normalized to [0, 1]
    sino_noisy : ndarray
        Noisy sparse-view sinogram
    angles_sparse : ndarray
        Array of sparse projection angles
    """
    # Generate Shepp-Logan phantom
    phantom = slp()
    if phantom.shape[0] != image_size:
        phantom = resize(phantom, (image_size, image_size), anti_aliasing=True)
    phantom = phantom.astype(np.float64)
    
    # Normalize to [0, 1]
    phantom = (phantom - phantom.min()) / (phantom.max() - phantom.min() + 1e-10)
    
    print(f"[DATA] Phantom shape: {phantom.shape}, range: [{phantom.min():.3f}, {phantom.max():.3f}]")
    
    # Sparse-view sinogram
    angles_sparse = np.linspace(0, 180, n_angles_sparse, endpoint=False)
    sino_sparse = radon(phantom, theta=angles_sparse, circle=True)
    
    # Add noise
    noise = np.random.randn(*sino_sparse.shape) * noise_level * sino_sparse.max()
    sino_noisy = sino_sparse + noise
    
    print(f"[DATA] Sinogram shape: {sino_noisy.shape} ({n_angles_sparse} angles)")
    
    return phantom, sino_noisy, angles_sparse
