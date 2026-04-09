import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.optimize import minimize

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

def run_inversion(data: dict) -> dict:
    """
    Run phase retrieval optimization.
    
    Uses L-BFGS-B optimization with multi-start strategy to find Zernike
    coefficients that best match the measured PSFs in amplitude domain.
    
    Returns dictionary with retrieved coefficients and phase.
    """
    pupil_mask = data['pupil_mask']
    npix = data['npix']
    n_pad = data['n_pad']
    pad_offset = data['pad_offset']
    retrieval_basis = data['retrieval_basis']
    diversity_phase_rad = data['diversity_phase_rad']
    psf_infocus_norm = data['psf_infocus_norm']
    psf_defocus_norm = data['psf_defocus_norm']
    n_modes = data['n_modes']
    
    call_count = [0]
    
    def objective(coefs_waves):
        """
        Cost function: sum of squared differences between model and measured
        PSF amplitudes (sqrt of intensities) for both diversity channels.
        """
        _, psf_m1, psf_m2 = forward_operator(
            coefs_waves, retrieval_basis, pupil_mask,
            npix, n_pad, pad_offset, diversity_phase_rad
        )
        
        # Channel 1: in-focus
        psf_m1_norm = psf_m1 / (psf_m1.sum() + 1e-30)
        err1 = np.sum((np.sqrt(psf_m1_norm) - np.sqrt(psf_infocus_norm))**2)
        
        # Channel 2: defocused
        psf_m2_norm = psf_m2 / (psf_m2.sum() + 1e-30)
        err2 = np.sum((np.sqrt(psf_m2_norm) - np.sqrt(psf_defocus_norm))**2)
        
        call_count[0] += 1
        return err1 + err2
    
    def gradient(coefs_waves, eps=5e-4):
        """Finite-difference gradient."""
        grad = np.zeros(n_modes)
        f0 = objective(coefs_waves)
        for i in range(n_modes):
            cp = coefs_waves.copy()
            cp[i] += eps
            grad[i] = (objective(cp) - f0) / eps
        return grad
    
    print("\nStarting optimization-based phase retrieval...")
    print("  Phase 1: Coarse search with L-BFGS-B...")
    
    # Multi-start optimization
    best_result = None
    best_cost = float('inf')
    
    for trial in range(5):
        if trial == 0:
            x_init = np.zeros(n_modes)
        else:
            x_init = np.random.randn(n_modes) * 0.1
        
        result = minimize(
            objective,
            x_init,
            method='L-BFGS-B',
            jac=gradient,
            options={'maxiter': 60, 'ftol': 1e-12, 'gtol': 1e-8},
        )
        
        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result
            print(f"  Trial {trial}: cost = {result.fun:.6e} (best so far)")
        else:
            print(f"  Trial {trial}: cost = {result.fun:.6e}")
    
    coefs_opt = best_result.x
    print(f"\n  Best cost after coarse: {best_cost:.6e}")
    
    # Fine refinement
    print("  Phase 2: Fine refinement...")
    result_fine = minimize(
        objective,
        coefs_opt,
        method='L-BFGS-B',
        jac=lambda c: gradient(c, eps=1e-5),
        options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10},
    )
    coefs_opt = result_fine.x
    print(f"  Final cost: {result_fine.fun:.6e}")
    print(f"  Total function evaluations: {call_count[0]}")
    
    # Compute retrieved phase
    retrieved_phase_rad, _, _ = forward_operator(
        coefs_opt, retrieval_basis, pupil_mask,
        npix, n_pad, pad_offset
    )
    
    return {
        'coefs_opt': coefs_opt,
        'retrieved_phase_rad': retrieved_phase_rad,
        'final_cost': result_fine.fun,
        'n_evaluations': call_count[0],
    }
