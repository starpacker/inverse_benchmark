import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.ndimage import gaussian_filter

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_endmember_spectra(n_bands, n_end, rng):
    """
    Create synthetic endmember spectra resembling mineral reflectances.
    """
    wavelengths = np.linspace(400, 2500, n_bands)
    E = np.zeros((n_bands, n_end))

    specs = [
        (0.8, [(550, 40, 0.55), (1200, 60, 0.50)]),
        (0.15, [(900, 50, 0.10)]),
        (0.5, [(480, 40, 0.40), (2200, 100, 0.45)]),
        (0.35, [(700, 60, 0.20), (1600, 100, 0.30)]),
    ]

    for i in range(n_end):
        base, features = specs[i % len(specs)]
        E[:, i] = base
        for center, width, depth in features:
            E[:, i] -= depth * np.exp(-(wavelengths - center)**2 / (2 * width**2))
        E[:, i] += 0.02 * rng.standard_normal(n_bands)
        E[:, i] = np.clip(E[:, i], 0.01, 1.0)

    return E, wavelengths

def generate_abundance_maps(img_size, n_end, rng):
    """
    Create spatially smooth, physically realistic abundance maps.
    """
    A = np.zeros((n_end, img_size, img_size))
    for i in range(n_end):
        raw = rng.standard_normal((img_size, img_size))
        A[i] = gaussian_filter(raw, sigma=8 + 2 * i)

    A_exp = np.exp(2.0 * (A - A.max(axis=0, keepdims=True)))
    A_sum = A_exp.sum(axis=0, keepdims=True)
    A_norm = A_exp / A_sum
    
    corners = [(0,0), (0,img_size-1), (img_size-1,0), (img_size-1,img_size-1)]
    for i in range(min(n_end, 4)):
        r, c = corners[i]
        A_norm[:, r, c] = 0.0
        A_norm[i, r, c] = 1.0

    return A_norm.reshape(n_end, -1)

def load_and_preprocess_data(n_bands, img_size, n_endmembers, snr_db, seed):
    """
    Generate synthetic hyperspectral data with known endmembers and abundances.
    
    Returns:
        dict containing:
            - Y_noisy: Observed mixed spectra (L x P)
            - Y_clean: Clean mixed spectra (L x P)
            - E_gt: Ground truth endmember matrix (L x R)
            - A_gt: Ground truth abundance matrix (R x P)
            - wavelengths: Wavelength array
            - rng: Random number generator for reproducibility
    """
    rng = np.random.default_rng(seed)
    n_pixels = img_size ** 2
    
    # Generate endmember spectra
    E_gt, wavelengths = generate_endmember_spectra(n_bands, n_endmembers, rng)
    
    # Generate abundance maps
    A_gt = generate_abundance_maps(img_size, n_endmembers, rng)
    
    # Generate noisy observations using forward model
    Y_clean = E_gt @ A_gt
    signal_power = np.mean(Y_clean ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    N = np.sqrt(noise_power) * rng.standard_normal(Y_clean.shape)
    Y_noisy = Y_clean + N
    
    print(f"  {n_bands} bands × {n_pixels} pixels, {n_endmembers} endmembers")
    print(f"  Y shape: {Y_noisy.shape}, SNR: {snr_db} dB")
    print(f"  Wavelength range: {wavelengths[0]:.0f}–{wavelengths[-1]:.0f} nm")
    
    return {
        'Y_noisy': Y_noisy,
        'Y_clean': Y_clean,
        'E_gt': E_gt,
        'A_gt': A_gt,
        'wavelengths': wavelengths,
        'rng': rng,
        'img_size': img_size,
        'n_endmembers': n_endmembers
    }
