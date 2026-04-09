import numpy as np

import matplotlib

matplotlib.use('Agg')

from skimage.transform import radon, iradon

def load_and_preprocess_data(nz=64, ny=128, nx=128, n_angles=180, photon_count=5e4, readout_std=0.3, seed=42):
    """
    Generate a synthetic 3D phantom for OPT reconstruction and compute noisy sinograms.
    
    The phantom contains:
    - A large outer ellipsoid (simulating a tissue sample)
    - Several internal structures (spheres, cylinders) with varying intensities
    
    Returns:
        phantom: 3D ground truth phantom array (nz, ny, nx)
        sinograms_noisy: 3D array of noisy sinograms (nz, sino_height, n_angles)
        theta: projection angles
        params: dictionary of parameters used
    """
    np.random.seed(seed)
    
    # Create 3D phantom
    phantom = np.zeros((nz, ny, nx), dtype=np.float64)
    
    cz, cy, cx = nz // 2, ny // 2, nx // 2
    
    # Create coordinate grids
    z, y, x = np.ogrid[:nz, :ny, :nx]
    
    # Outer ellipsoid (normalized coords)
    outer = ((z - cz) / (cz * 0.85))**2 + ((y - cy) / (cy * 0.8))**2 + ((x - cx) / (cx * 0.8))**2
    phantom[outer <= 1.0] = 0.5
    
    # Inner sphere 1 (high intensity, offset)
    r1 = ((z - cz) / 12.0)**2 + ((y - (cy - 20)) / 15.0)**2 + ((x - (cx + 15)) / 15.0)**2
    phantom[r1 <= 1.0] = 1.0
    
    # Inner sphere 2 (medium intensity, opposite side)
    r2 = ((z - cz) / 10.0)**2 + ((y - (cy + 25)) / 12.0)**2 + ((x - (cx - 20)) / 12.0)**2
    phantom[r2 <= 1.0] = 0.8
    
    # Small dense sphere near center
    r3 = ((z - cz) / 6.0)**2 + ((y - cy) / 8.0)**2 + ((x - cx) / 8.0)**2
    phantom[r3 <= 1.0] = 0.9
    
    # A rod/cylinder along z-axis (off-center)
    rod = ((y - (cy + 10)) / 5.0)**2 + ((x - (cx - 30)) / 5.0)**2
    phantom[:, rod[0, :, :] <= 1.0] = 0.7
    
    # Another small feature
    r4 = ((z - (cz + 15)) / 8.0)**2 + ((y - (cy - 15)) / 6.0)**2 + ((x - (cx + 30)) / 6.0)**2
    phantom[r4 <= 1.0] = 0.6
    
    # Compute projection angles
    theta = np.linspace(0., 180., n_angles, endpoint=False)
    
    # Compute sinograms (forward projection)
    test_sino = radon(phantom[0], theta=theta, circle=True)
    sino_height = test_sino.shape[0]
    sinograms = np.zeros((nz, sino_height, n_angles))
    
    for z_idx in range(nz):
        sinograms[z_idx] = radon(phantom[z_idx], theta=theta, circle=True)
    
    # Add noise to sinograms
    sinograms_noisy = np.zeros_like(sinograms)
    for z_idx in range(nz):
        sinogram = sinograms[z_idx]
        sino_min = sinogram.min()
        sino_max = sinogram.max()
        
        if sino_max - sino_min < 1e-10:
            sinograms_noisy[z_idx] = sinogram.copy()
        else:
            sino_norm = (sinogram - sino_min) / (sino_max - sino_min)
            
            # Poisson noise: scale to photon counts, sample, scale back
            sino_photons = sino_norm * photon_count
            sino_noisy = np.random.poisson(sino_photons).astype(np.float64) / photon_count
            
            # Readout noise (Gaussian)
            sino_noisy += np.random.normal(0, readout_std / photon_count, sino_noisy.shape)
            
            # Scale back to original range
            sino_noisy = sino_noisy * (sino_max - sino_min) + sino_min
            sinograms_noisy[z_idx] = sino_noisy
    
    params = {
        'nz': nz,
        'ny': ny,
        'nx': nx,
        'n_angles': n_angles,
        'photon_count': photon_count,
        'readout_std': readout_std,
        'seed': seed
    }
    
    return phantom, sinograms_noisy, theta, params
