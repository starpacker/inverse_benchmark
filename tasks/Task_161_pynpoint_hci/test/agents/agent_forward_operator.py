import numpy as np

from scipy.ndimage import rotate as ndrotate

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

def forward_operator(
    planet_image,
    angles,
    star_flux=1e5,
    psf_fwhm=5.0,
    speckle_amp=0.03,
    read_noise=8.0,
    seed=42
):
    """
    Forward model: True scene → ADI observation cube.
    
    Given a clean planet image (in the reference frame), this function:
    1. Adds stellar PSF at center
    2. Adds broad stellar halo
    3. Adds quasi-static speckle pattern
    4. Rotates planet by parallactic angles
    5. Adds photon noise (Poisson) and read noise (Gaussian)
    
    Parameters
    ----------
    planet_image : ndarray (image_size, image_size)
        Clean planet image in reference frame
    angles : ndarray (n_frames,)
        Parallactic angles in degrees
    star_flux : float
        Peak stellar flux
    psf_fwhm : float
        PSF full-width half-maximum in pixels
    speckle_amp : float
        Amplitude of quasi-static speckle
    read_noise : float
        Gaussian read noise standard deviation
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    cube : ndarray (n_frames, image_size, image_size)
        Simulated ADI observation cube
    """
    rng = np.random.default_rng(seed)
    n_frames = len(angles)
    image_size = planet_image.shape[0]
    cx = cy = image_size // 2
    
    # Stellar PSF (centred, constant across frames)
    star_psf = gaussian_2d(image_size, cx, cy, star_flux, psf_fwhm)

    # Broad stellar halo
    y, x = np.mgrid[:image_size, :image_size]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) + 1e-3
    halo = star_flux * 0.005 / (1.0 + (r / 8.0) ** 2)

    # Quasi-static speckle pattern
    speckle = make_speckle_field(image_size, rng, amp=speckle_amp)
    speckle_pattern = halo * speckle
    
    # Static component (star + halo + speckle)
    static_component = star_psf + halo + speckle_pattern

    # Build data cube
    cube = np.zeros((n_frames, image_size, image_size))
    for i, angle in enumerate(angles):
        # Rotate planet image by parallactic angle
        rotated_planet = ndrotate(planet_image, angle, reshape=False, order=3)
        
        # Combine static component with rotated planet
        frame = static_component + rotated_planet

        # Photon noise (Poisson) + read noise (Gaussian)
        frame_pos = np.clip(frame, 0.0, None)
        noisy = rng.poisson(frame_pos).astype(np.float64)
        noisy += rng.normal(0, read_noise, frame.shape)
        cube[i] = noisy
    
    return cube
