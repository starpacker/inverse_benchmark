import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.fft import fft2, ifft2, fftshift, ifftshift

def load_and_preprocess_data(obj_size, det_size, snr_db, seed):
    """
    Generate synthetic CDI data: complex-valued object and its diffraction pattern.
    
    Returns:
        obj_gt: Ground truth complex object
        support_gt: Binary support mask
        intensity_noisy: Noisy diffraction intensity
        intensity_clean: Clean diffraction intensity
        rng: Random number generator for later use
    """
    rng = np.random.default_rng(seed)
    
    # Generate complex-valued object with amplitude and phase structure
    amp = np.zeros((det_size, det_size))
    cx, cy = det_size // 2, det_size // 2
    Y, X = np.mgrid[:det_size, :det_size]

    # Crystal-like shape
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    amp[r < obj_size // 3] = 0.8

    # Internal structure
    amp[(np.abs(X - cx) < obj_size // 6) &
        (np.abs(Y - cy) < obj_size // 4)] = 1.0

    # Small features
    amp[(X - cx + 10)**2 + (Y - cy + 8)**2 < 16] = 0.6
    amp[(X - cx - 8)**2 + (Y - cy - 12)**2 < 9] = 0.7

    # Phase: smooth strain field
    phase = np.zeros((det_size, det_size))
    phase = 0.5 * np.sin(2 * np.pi * (X - cx) / obj_size) * (amp > 0)
    phase += 0.3 * np.cos(2 * np.pi * (Y - cy) / (obj_size * 0.8)) * (amp > 0)

    obj_gt = amp * np.exp(1j * phase)
    support_gt = amp > 0.01

    # Compute far-field diffraction intensity pattern
    F_obj = fftshift(fft2(ifftshift(obj_gt)))
    intensity_clean = np.abs(F_obj)**2

    # Normalise
    intensity_clean = intensity_clean / intensity_clean.max()

    # Poisson-like noise
    sig_power = np.mean(intensity_clean**2)
    noise_power = sig_power / (10**(snr_db / 10))
    noise = np.sqrt(noise_power) * rng.standard_normal(intensity_clean.shape)
    intensity_noisy = np.maximum(intensity_clean + noise, 0)

    return obj_gt, support_gt, intensity_noisy, intensity_clean, rng
