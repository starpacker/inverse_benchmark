import warnings

warnings.filterwarnings('ignore')

import numpy as np

import matplotlib

matplotlib.use('Agg')

from skimage.data import shepp_logan_phantom

from skimage.transform import radon, iradon, resize

def load_and_preprocess_data(image_size, num_angles, snr_db, random_seed=42):
    """
    Load and preprocess data for CT reconstruction.
    
    Generates Shepp-Logan phantom, computes its Radon transform (sinogram),
    and adds Gaussian noise.
    
    Args:
        image_size: Resolution of the phantom image
        num_angles: Number of projection angles
        snr_db: Target signal-to-noise ratio in dB
        random_seed: Random seed for reproducibility
        
    Returns:
        phantom: Ground truth image (2D numpy array)
        sinogram_noisy: Noisy sinogram (2D numpy array)
        sinogram_clean: Clean sinogram (2D numpy array)
        theta: Array of projection angles
        actual_snr: Actual SNR achieved after adding noise
    """
    np.random.seed(random_seed)
    
    # Generate Shepp-Logan phantom
    phantom_full = shepp_logan_phantom()
    phantom = resize(phantom_full, (image_size, image_size), anti_aliasing=True)
    phantom = phantom.astype(np.float64)
    
    # Compute projection angles
    theta = np.linspace(0, 179, num_angles, endpoint=True)
    
    # Forward model: Radon transform
    sinogram_clean = radon(phantom, theta=theta, circle=True)
    
    # Add Gaussian noise to achieve target SNR
    signal_power = np.mean(sinogram_clean ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), sinogram_clean.shape)
    sinogram_noisy = sinogram_clean + noise
    
    # Compute actual SNR
    actual_snr = 10 * np.log10(np.mean(sinogram_clean ** 2) /
                                np.mean((sinogram_noisy - sinogram_clean) ** 2))
    
    return phantom, sinogram_noisy, sinogram_clean, theta, actual_snr
