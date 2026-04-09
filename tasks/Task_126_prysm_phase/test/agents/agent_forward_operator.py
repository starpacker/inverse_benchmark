import numpy as np

import matplotlib

matplotlib.use('Agg')

from prysm.polynomials import zernike_nm, zernike_nm_sequence, sum_of_2d_modes

def forward_operator(
    coefs_waves: np.ndarray,
    retrieval_basis: np.ndarray,
    pupil_mask: np.ndarray,
    npix: int,
    n_pad: int,
    pad_offset: int,
    diversity_phase_rad: np.ndarray = None
) -> tuple:
    """
    Forward model: Convert Zernike coefficients to PSF(s).
    
    Given Zernike coefficients (in waves), compute:
    1. Phase map (radians)
    2. In-focus PSF
    3. Defocused PSF (if diversity_phase_rad is provided)
    
    Returns (phase_rad, psf_infocus, psf_defocus) where psf_defocus is None
    if diversity_phase_rad is not provided.
    """
    # Convert coefficients to phase
    phase_waves = sum_of_2d_modes(retrieval_basis, coefs_waves)
    phase_rad = phase_waves * pupil_mask * 2 * np.pi
    
    def make_psf(phase):
        """Generate PSF from phase."""
        E_pupil = pupil_mask * np.exp(1j * phase)
        E_pad = np.zeros((n_pad, n_pad), dtype=complex)
        E_pad[pad_offset:pad_offset+npix, pad_offset:pad_offset+npix] = E_pupil
        E_focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_pad), norm='ortho'))
        return np.abs(E_focal)**2
    
    # In-focus PSF
    psf_infocus = make_psf(phase_rad)
    
    # Defocused PSF
    psf_defocus = None
    if diversity_phase_rad is not None:
        psf_defocus = make_psf(phase_rad + diversity_phase_rad)
    
    return phase_rad, psf_infocus, psf_defocus
