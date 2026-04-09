import numpy as np

import matplotlib

matplotlib.use("Agg")

def gaussian_2d(size, cx, cy, flux, fwhm):
    """2-D Gaussian centred at (cx, cy) with peak = `flux`."""
    sigma = fwhm / 2.355
    y, x = np.mgrid[:size, :size]
    return flux * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2))

def make_speckle_field(size, rng, n_modes=25, amp=0.03):
    """
    Quasi-static speckle field as a sum of random sinusoidal modes.
    Fixed in the pupil plane → identical in every frame.
    Returns a zero-mean, amplitude-controlled field.
    """
    field = np.zeros((size, size))
    yy, xx = np.mgrid[:size, :size]
    for _ in range(n_modes):
        kx = rng.uniform(-0.3, 0.3)
        ky = rng.uniform(-0.3, 0.3)
        phase = rng.uniform(0, 2 * np.pi)
        a = rng.uniform(0.3, 1.0)
        field += a * np.cos(2 * np.pi * (kx * xx + ky * yy) + phase)
    field -= field.mean()
    field *= amp / (field.std() + 1e-10)
    return field

def load_and_preprocess_data(
    n_frames=100,
    image_size=101,
    planet_sep=30,
    planet_contrast=1e-2,
    total_rotation=90.0,
    star_flux=1e5,
    psf_fwhm=5.0,
    planet_fwhm=5.0,
    read_noise=8.0,
    speckle_amp=0.03,
    seed=42
):
    """
    Synthesize and preprocess an ADI data cube.
    
    This function generates synthetic high-contrast imaging data including:
    - Star PSF (central bright source)
    - Broad stellar halo
    - Quasi-static speckle pattern
    - Planet signal at specified separation and contrast
    - Photon noise (Poisson) and read noise (Gaussian)
    
    Returns
    -------
    data_dict : dict containing:
        - 'cube': (n_frames, image_size, image_size) ADI data cube
        - 'angles': (n_frames,) parallactic angles in degrees
        - 'ground_truth': dict with planet position, flux, and clean image
        - 'params': dict with simulation parameters
    """
    rng = np.random.default_rng(seed)
    cx = cy = image_size // 2
    angles = np.linspace(-total_rotation / 2, total_rotation / 2, n_frames)

    # Stellar PSF (centred, constant across frames)
    star_psf = gaussian_2d(image_size, cx, cy, star_flux, psf_fwhm)

    # Broad stellar halo (Moffat-like, extends to large radii)
    y, x = np.mgrid[:image_size, :image_size]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) + 1e-3
    halo = star_flux * 0.005 / (1.0 + (r / 8.0) ** 2)

    # Quasi-static speckle: modulation of halo
    speckle = make_speckle_field(image_size, rng, amp=speckle_amp)
    speckle_pattern = halo * speckle

    # Planet (reference frame: angle = 0, planet along +x axis)
    planet_flux = star_flux * planet_contrast
    planet_row_ref = cy
    planet_col_ref = cx + planet_sep
    clean_planet = gaussian_2d(image_size, planet_col_ref, planet_row_ref,
                               planet_flux, planet_fwhm)

    # Build data cube
    cube = np.zeros((n_frames, image_size, image_size))
    for i, angle in enumerate(angles):
        frame = star_psf + halo + speckle_pattern  # fixed component

        # Planet rotates with parallactic angle
        angle_rad = np.radians(angle)
        px = cx + planet_sep * np.cos(angle_rad)
        py = cy + planet_sep * np.sin(angle_rad)
        planet_img = gaussian_2d(image_size, px, py, planet_flux, planet_fwhm)
        frame = frame + planet_img

        # Photon noise (Poisson) + read noise (Gaussian)
        frame_pos = np.clip(frame, 0.0, None)
        noisy = rng.poisson(frame_pos).astype(np.float64)
        noisy += rng.normal(0, read_noise, frame.shape)
        cube[i] = noisy

    ground_truth = {
        "planet_position": (planet_row_ref, planet_col_ref),
        "planet_flux": planet_flux,
        "clean_planet_image": clean_planet,
    }
    
    params = {
        "n_frames": n_frames,
        "image_size": image_size,
        "planet_sep": planet_sep,
        "planet_contrast": planet_contrast,
        "total_rotation": total_rotation,
        "star_flux": star_flux,
        "psf_fwhm": psf_fwhm,
        "planet_fwhm": planet_fwhm,
        "read_noise": read_noise,
        "speckle_amp": speckle_amp,
    }
    
    data_dict = {
        "cube": cube,
        "angles": angles,
        "ground_truth": ground_truth,
        "params": params,
    }
    
    return data_dict
