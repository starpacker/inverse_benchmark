import numpy as np

import matplotlib

matplotlib.use('Agg')

from skimage.data import shepp_logan_phantom

from skimage.transform import resize, radon, iradon

def load_and_preprocess_data(image_size=256, n_angles=180, noise_level=0.01, seed=42):
    """
    Load and preprocess data for CT reconstruction.
    
    Parameters:
    -----------
    image_size : int
        Size of the phantom image (N x N)
    n_angles : int
        Number of projection angles
    noise_level : float
        Noise standard deviation as fraction of sinogram max
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict containing:
        - 'ground_truth': Ground truth phantom image
        - 'sinogram_noisy': Noisy sinogram measurement
        - 'sinogram_clean': Clean sinogram (for reference)
        - 'theta_angles': Projection angles
        - 'noise_std': Standard deviation of added noise
    """
    # Generate Shepp-Logan phantom
    gt = resize(shepp_logan_phantom(), (image_size, image_size), anti_aliasing=True).astype('float64')
    gt = (gt - gt.min()) / (gt.max() - gt.min())  # normalize to [0, 1]
    
    print(f"[INFO] Phantom shape: {gt.shape}, range: [{gt.min():.4f}, {gt.max():.4f}]")
    
    # Define projection angles
    theta_angles = np.linspace(0., 180., n_angles, endpoint=False)
    
    # Compute clean sinogram
    sinogram_clean = radon(gt, theta=theta_angles, circle=False)
    
    # Add Gaussian noise
    noise_std = noise_level * sinogram_clean.max()
    rng = np.random.default_rng(seed)
    noise = noise_std * rng.standard_normal(sinogram_clean.shape)
    sinogram_noisy = sinogram_clean + noise
    
    print(f"[INFO] Sinogram shape: {sinogram_noisy.shape}")
    print(f"[INFO] Noise std: {noise_std:.4f} ({noise_level*100:.0f}% of sinogram max {sinogram_clean.max():.2f})")
    
    return {
        'ground_truth': gt,
        'sinogram_noisy': sinogram_noisy,
        'sinogram_clean': sinogram_clean,
        'theta_angles': theta_angles,
        'noise_std': noise_std
    }
