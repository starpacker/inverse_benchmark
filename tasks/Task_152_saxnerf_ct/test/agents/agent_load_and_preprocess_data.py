import numpy as np

import matplotlib

matplotlib.use('Agg')

from skimage.data import shepp_logan_phantom

from skimage.transform import radon, iradon, resize

def load_and_preprocess_data(size=128, n_full_angles=180, n_sparse_angles=30):
    """
    Generate Shepp-Logan phantom and compute sinograms at full and sparse angles.
    
    Returns:
        phantom: Ground truth image (size x size)
        sinogram_full: Full sinogram (n_detectors x n_full_angles)
        sinogram_sparse: Sparse sinogram (n_detectors x n_sparse_angles)
        angles_full: Full angle array
        angles_sparse: Sparse angle array
        config: Dictionary with configuration parameters
    """
    # Generate Shepp-Logan phantom
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (size, size), anti_aliasing=True)
    # Normalize to [0, 1]
    phantom = (phantom - phantom.min()) / (phantom.max() - phantom.min() + 1e-12)
    
    # Generate angle arrays
    angles_full = np.linspace(0, 180, n_full_angles, endpoint=False)
    angles_sparse = np.linspace(0, 180, n_sparse_angles, endpoint=False)
    
    # Compute sinograms via Radon transform
    sinogram_full = radon(phantom, theta=angles_full, circle=True)
    sinogram_sparse = radon(phantom, theta=angles_sparse, circle=True)
    
    config = {
        'size': size,
        'n_full_angles': n_full_angles,
        'n_sparse_angles': n_sparse_angles
    }
    
    return phantom, sinogram_full, sinogram_sparse, angles_full, angles_sparse, config
