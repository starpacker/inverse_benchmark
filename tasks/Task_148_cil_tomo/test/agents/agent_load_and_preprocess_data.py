import matplotlib

matplotlib.use('Agg')

import os

import numpy as np

from skimage.data import shepp_logan_phantom

from skimage.transform import radon, iradon, resize

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

def load_and_preprocess_data(size=128, n_angles=90, noise_level=0.02):
    """
    Generate a Shepp-Logan phantom, compute the forward projection (sinogram),
    and add noise to simulate realistic CT acquisition.
    
    Returns:
        phantom: Ground truth image (size x size)
        sinogram_noisy: Noisy sinogram
        sinogram_clean: Clean sinogram (for SNR calculation)
        theta: Projection angles
        adjoint_scale: Calibrated adjoint scaling factor
    """
    # Generate Shepp-Logan phantom
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (size, size), anti_aliasing=True)
    phantom = (phantom - phantom.min()) / (phantom.max() - phantom.min() + 1e-12)
    phantom = phantom.astype(np.float64)
    
    print(f"Phantom shape: {phantom.shape}, "
          f"range: [{phantom.min():.3f}, {phantom.max():.3f}]")
    
    # Define projection angles (limited angle scenario)
    theta = np.linspace(0, 180, n_angles, endpoint=False)
    
    # Forward projection (Radon transform)
    sinogram_clean = radon(phantom, theta=theta, circle=True)
    print(f"Sinogram shape: {sinogram_clean.shape}")
    
    # Add Gaussian noise
    noise = noise_level * np.random.randn(*sinogram_clean.shape)
    sinogram_noisy = sinogram_clean + noise
    
    snr_sino = 10 * np.log10(np.sum(sinogram_clean**2) /
                              (np.sum(noise**2) + 1e-30))
    print(f"Sinogram SNR: {snr_sino:.1f} dB")
    
    # Calibrate adjoint operator
    # iradon(filter=None) doesn't produce the exact adjoint of radon().
    # We empirically determine the scaling factor via the dot-product test.
    rng = np.random.RandomState(999)
    x_test = rng.randn(size, size)
    Ax = radon(x_test, theta=theta, circle=True)
    y_test = rng.randn(*Ax.shape)
    ATy = iradon(y_test, theta=theta, output_size=size, filter_name=None, circle=True)
    
    lhs = np.sum(Ax * y_test)
    rhs = np.sum(x_test * ATy)
    adjoint_scale = lhs / rhs
    print(f"Adjoint scale factor: {adjoint_scale:.4f}")
    
    return phantom, sinogram_noisy, sinogram_clean, theta, adjoint_scale, noise_level, n_angles
