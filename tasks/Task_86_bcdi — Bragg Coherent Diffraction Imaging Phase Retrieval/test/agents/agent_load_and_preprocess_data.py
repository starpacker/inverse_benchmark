import numpy as np

import matplotlib

matplotlib.use('Agg')

import warnings

from scipy.ndimage import gaussian_filter

warnings.filterwarnings('ignore')

def load_and_preprocess_data(N=64, snr_db=30):
    """
    Generate synthetic nanocrystal and compute noisy diffraction intensity.
    
    Returns:
        intensity_noisy: 3D noisy diffraction intensity
        obj_true: 3D complex ground truth object
        support: 3D binary support mask
        config: dictionary of configuration parameters
    """
    np.random.seed(42)
    
    # Create a faceted nanocrystal support using truncated octahedron shape
    x = np.linspace(-1, 1, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    # Sphere base shape
    crystal_radius = 0.4
    
    # Truncated octahedron: intersection of cube and octahedron
    cube_mask = (np.abs(X) < crystal_radius * 1.2) & \
                (np.abs(Y) < crystal_radius * 1.2) & \
                (np.abs(Z) < crystal_radius * 1.2)
    octa_mask = (np.abs(X) + np.abs(Y) + np.abs(Z)) < crystal_radius * 2.0
    
    support = (cube_mask & octa_mask).astype(float)
    
    # Smooth the edges slightly
    support_smooth = gaussian_filter(support, sigma=0.8)
    support = (support_smooth > 0.5).astype(float)
    
    # Create strain field (smooth phase variation inside crystal)
    phase = 1.5 * (0.3*X**2 - 0.2*Y**2 + 0.1*Z**2 + 0.4*X*Y + 0.2*X - 0.1*Z)
    
    # Create complex object: varying amplitude (density) + phase (strain)
    amplitude = support * (0.7 + 0.3 * np.exp(-(X**2 + Y**2 + Z**2) / (crystal_radius**2)))
    obj_true = amplitude * np.exp(1j * phase)
    
    print(f"  Object shape: {obj_true.shape}")
    print(f"  Support voxels: {np.sum(support > 0)} / {N**3}")
    print(f"  Phase range: [{phase[support > 0].min():.3f}, {phase[support > 0].max():.3f}] rad")
    print(f"  Oversampling ratio: {N**3 / np.sum(support > 0):.1f}")
    
    # Compute clean diffraction intensity
    obj_ft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj_true)))
    intensity_clean = np.abs(obj_ft)**2
    
    print(f"[FWD] Intensity range: [{intensity_clean.min():.2f}, {intensity_clean.max():.2f}]")
    print(f"[FWD] Dynamic range: {intensity_clean.max()/intensity_clean[intensity_clean>0].min():.1f}")
    
    # Add Poisson-like noise
    total_counts = 1e6
    scale = total_counts / intensity_clean.sum()
    intensity_scaled = intensity_clean * scale
    
    noisy = np.random.poisson(np.maximum(intensity_scaled, 0)).astype(float)
    intensity_noisy = noisy / scale
    
    actual_snr = 10 * np.log10(np.sum(intensity_clean**2) /
                                np.sum((intensity_clean - intensity_noisy)**2))
    print(f"  Target SNR: {snr_db} dB, Actual SNR: {actual_snr:.1f} dB")
    
    config = {
        'N': N,
        'snr_db': snr_db,
        'beta': 0.9,
        'n_er': 50,
        'n_hio': 200,
        'n_cycles': 5
    }
    
    return intensity_noisy, obj_true, support, config
