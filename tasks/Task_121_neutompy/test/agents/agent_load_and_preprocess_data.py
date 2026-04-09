import numpy as np

import matplotlib

matplotlib.use("Agg")

from skimage.transform import radon, iradon

def load_and_preprocess_data(n, n_angles, noise_level, i0, seed):
    """
    Create ground truth phantom and simulate neutron transmission radiography.
    
    This function:
    1. Creates a 2D attenuation coefficient phantom
    2. Computes ideal sinogram via Radon transform
    3. Applies Beer-Lambert noise model (Poisson + readout noise)
    
    Parameters
    ----------
    n : int
        Image size (n x n pixels)
    n_angles : int
        Number of projection angles
    noise_level : float
        Noise level parameter (not directly used, noise is Poisson-based)
    i0 : float
        Incident neutron flux (counts)
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'phantom': ground truth attenuation map
        - 'sinogram_noisy': noisy sinogram for reconstruction
        - 'sinogram_ideal': ideal sinogram
        - 'angles': projection angles in degrees
        - 'i0': incident flux
    """
    np.random.seed(seed)
    
    # Create phantom
    phantom = np.zeros((n, n), dtype=np.float64)
    yy, xx = np.ogrid[:n, :n]
    c = n // 2

    # Scale factor: physical_mu * pixel_size
    s = 0.04  # pixel size in cm

    # Outer aluminum casing (cylinder)
    r_outer = np.sqrt((yy - c)**2 + (xx - c)**2)
    phantom[r_outer < 0.42 * n] = 0.1 * s   # aluminum matrix

    # Steel cylinder 1 (off-center)
    r1 = np.sqrt((yy - c + 30)**2 + (xx - c - 40)**2)
    phantom[r1 < 25] = 1.0 * s

    # Steel cylinder 2
    r2 = np.sqrt((yy - c - 35)**2 + (xx - c + 25)**2)
    phantom[r2 < 20] = 1.0 * s

    # Water-filled cavity (high attenuation for neutrons)
    r3 = np.sqrt((yy - c + 10)**2 + (xx - c + 10)**2)
    phantom[r3 < 15] = 3.5 * s

    # Lead inclusion (moderate attenuation)
    r4 = np.sqrt((yy - c - 20)**2 + (xx - c - 30)**2)
    phantom[r4 < 12] = 0.4 * s

    # Small void (crack/pore)
    r5 = np.sqrt((yy - c + 40)**2 + (xx - c + 50)**2)
    phantom[r5 < 8] = 0.0

    # Another small steel rod
    r6 = np.sqrt((yy - c - 50)**2 + (xx - c - 10)**2)
    phantom[r6 < 10] = 0.8 * s

    # Annular structure
    r7 = np.sqrt((yy - c + 45)**2 + (xx - c - 15)**2)
    phantom[(r7 > 12) & (r7 < 18)] = 1.2 * s

    # Compute projection angles
    angles = np.linspace(0, 180, n_angles, endpoint=False)

    # Radon transform → sinogram (ideal line integrals of μ)
    sinogram_ideal = radon(phantom, theta=angles, circle=False)

    # Beer-Lambert noise model:
    # I = I0 * exp(-sinogram) → Poisson noise → -ln(I_noisy / I0)
    transmitted = i0 * np.exp(-sinogram_ideal)
    transmitted_noisy = np.random.poisson(
        np.maximum(transmitted, 1).astype(np.float64)
    ).astype(np.float64)
    # Add Gaussian readout noise (small)
    transmitted_noisy += np.random.normal(0, 2.0, transmitted_noisy.shape)
    transmitted_noisy = np.maximum(transmitted_noisy, 1.0)  # avoid log(0)
    sinogram_noisy = -np.log(transmitted_noisy / i0)

    return {
        'phantom': phantom,
        'sinogram_noisy': sinogram_noisy,
        'sinogram_ideal': sinogram_ideal,
        'angles': angles,
        'i0': i0
    }
