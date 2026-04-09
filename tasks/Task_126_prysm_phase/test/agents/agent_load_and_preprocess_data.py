import numpy as np

import matplotlib

matplotlib.use('Agg')

from prysm.coordinates import make_xy_grid, cart_to_polar

from prysm.geometry import circle

from prysm.polynomials import zernike_nm, zernike_nm_sequence, sum_of_2d_modes

def load_and_preprocess_data(
    npix: int,
    wavelength: float,
    epd: float,
    q: int,
    photon_flux: float,
    readout_noise: float,
    defocus_waves: float,
    zernike_specs: list,
    rng_seed: int
) -> dict:
    """
    Load and preprocess data for phase retrieval.
    
    Creates coordinate grids, pupil mask, ground truth phase from Zernike specs,
    diversity defocus phase, and generates noisy PSF measurements.
    
    Returns a dictionary containing all necessary data for the inversion.
    """
    np.random.seed(rng_seed)
    
    # Create coordinate grids
    x, y = make_xy_grid(npix, diameter=epd)
    r, t = cart_to_polar(x, y)
    
    pupil_radius = epd / 2.0
    pupil_mask = circle(pupil_radius, r).astype(float)
    pupil_bool = pupil_mask > 0.5
    
    r_norm = r / pupil_radius
    
    # Ground truth phase from Zernike coefficients
    nms_truth = [(n, m) for n, m, _ in zernike_specs]
    coefs_truth_waves = np.array([c for _, _, c in zernike_specs])
    
    basis_truth = list(zernike_nm_sequence(nms_truth, r_norm, t, norm=True))
    basis_truth = np.array(basis_truth)
    
    true_phase_waves = sum_of_2d_modes(basis_truth, coefs_truth_waves)
    true_phase_waves *= pupil_mask
    true_phase_rad = true_phase_waves * 2 * np.pi
    
    # Diversity defocus phase
    defocus_mode = zernike_nm(2, 0, r_norm, t, norm=True)
    diversity_phase_rad = defocus_waves * defocus_mode * pupil_mask * 2 * np.pi
    
    # PSF generation parameters
    n_pad = npix * q
    pad_offset = (n_pad - npix) // 2
    
    def make_psf_local(phase_rad):
        """Generate PSF from phase."""
        E_pupil = pupil_mask * np.exp(1j * phase_rad)
        E_pad = np.zeros((n_pad, n_pad), dtype=complex)
        E_pad[pad_offset:pad_offset+npix, pad_offset:pad_offset+npix] = E_pupil
        E_focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_pad), norm='ortho'))
        return np.abs(E_focal)**2
    
    def add_noise_local(psf, pf, rn):
        """Add Poisson + readout noise."""
        psf_photons = psf / (psf.sum() + 1e-30) * pf
        noisy = np.random.poisson(np.clip(psf_photons, 0, None).astype(float)).astype(float)
        noisy += np.random.normal(0, rn, noisy.shape)
        return np.clip(noisy, 0, None)
    
    # Generate measurements
    psf_infocus = make_psf_local(true_phase_rad)
    psf_infocus_noisy = add_noise_local(psf_infocus, photon_flux, readout_noise)
    
    psf_defocus = make_psf_local(true_phase_rad + diversity_phase_rad)
    psf_defocus_noisy = add_noise_local(psf_defocus, photon_flux, readout_noise)
    
    # Normalize measurements
    total_1 = psf_infocus_noisy.sum()
    total_2 = psf_defocus_noisy.sum()
    psf_infocus_norm = psf_infocus_noisy / total_1
    psf_defocus_norm = psf_defocus_noisy / total_2
    
    # Build retrieval Zernike basis
    retrieval_nms = []
    for n in range(2, 6):
        for m in range(-n, n+1, 2):
            retrieval_nms.append((n, m))
    
    n_modes = len(retrieval_nms)
    
    retrieval_basis = list(zernike_nm_sequence(retrieval_nms, r_norm, t, norm=True))
    retrieval_basis = np.array(retrieval_basis)
    
    print(f"True phase: PV = {np.ptp(true_phase_rad[pupil_bool]):.3f} rad "
          f"({np.ptp(true_phase_waves[pupil_bool]):.3f} waves)")
    print(f"True phase RMS = {np.std(true_phase_rad[pupil_bool]):.3f} rad "
          f"({np.std(true_phase_waves[pupil_bool]):.3f} waves)")
    print(f"\nPSF shape: {psf_infocus.shape}")
    print(f"In-focus SNR ~ {np.sqrt(photon_flux):.0f}")
    print(f"\nRetrieval: {n_modes} Zernike modes (n=2..5)")
    
    return {
        'npix': npix,
        'q': q,
        'pupil_mask': pupil_mask,
        'pupil_bool': pupil_bool,
        'true_phase_rad': true_phase_rad,
        'diversity_phase_rad': diversity_phase_rad,
        'psf_infocus_norm': psf_infocus_norm,
        'psf_defocus_norm': psf_defocus_norm,
        'psf_infocus_noisy': psf_infocus_noisy,
        'psf_defocus_noisy': psf_defocus_noisy,
        'retrieval_basis': retrieval_basis,
        'retrieval_nms': retrieval_nms,
        'n_modes': n_modes,
        'zernike_specs': zernike_specs,
        'n_pad': n_pad,
        'pad_offset': pad_offset,
    }
