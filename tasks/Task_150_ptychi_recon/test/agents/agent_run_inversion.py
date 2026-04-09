import matplotlib

matplotlib.use('Agg')

import os

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def run_inversion(patterns, positions, probe_init, obj_shape,
                  n_iter=150, alpha_obj=1.0, alpha_probe=1.0,
                  probe_update_start=0):
    """
    Extended Ptychographic Iterative Engine (ePIE) reconstruction.

    Uses the classic max-denominator update:
        obj  += alpha_o * conj(P) / max|P|^2 * delta
        probe += alpha_p * conj(O) / max|O|^2 * delta
    with position randomisation each epoch.

    Parameters
    ----------
    patterns : list of ndarray
        Measured diffraction intensity patterns.
    positions : list of tuples
        List of (py, px) scan positions.
    probe_init : ndarray (complex)
        Initial estimate of the probe.
    obj_shape : tuple
        Shape of the object to reconstruct.
    n_iter : int
        Number of iterations.
    alpha_obj : float
        Step size for object update.
    alpha_probe : float
        Step size for probe update.
    probe_update_start : int
        Iteration to start probe updates (for stability).

    Returns
    -------
    dict
        Dictionary containing:
        - 'object': reconstructed complex object
        - 'probe': reconstructed complex probe
        - 'errors': list of error values per iteration
    """
    ph, pw = probe_init.shape
    obj = np.ones(obj_shape, dtype=np.complex128)
    probe = probe_init.copy()
    n_pos = len(positions)
    errors = []

    for it in range(n_iter):
        order = np.random.permutation(n_pos)
        err = 0.0
        for idx in order:
            py, px = positions[idx]
            meas = patterns[idx]

            # Extract current object patch
            patch = obj[py:py+ph, px:px+pw].copy()
            # Compute exit wave
            psi = probe * patch

            # Forward propagate to Fourier space
            PSI = np.fft.fft2(psi)
            # Apply modulus constraint
            amp_m = np.sqrt(np.maximum(meas, 0))
            amp_c = np.abs(PSI) + 1e-30
            PSI_new = amp_m * PSI / amp_c
            err += np.sum((amp_c - amp_m)**2)

            # Back propagate to real space
            psi_new = np.fft.ifft2(PSI_new)
            delta = psi_new - psi

            # Object update using ePIE formula
            P2max = np.max(np.abs(probe)**2) + 1e-12
            obj[py:py+ph, px:px+pw] += (
                alpha_obj * np.conj(probe) / P2max * delta)

            # Probe update (delayed start for stability)
            if it >= probe_update_start:
                O2max = np.max(np.abs(patch)**2) + 1e-12
                probe += alpha_probe * np.conj(patch) / O2max * delta

        errors.append(err / n_pos)
        if (it + 1) % 20 == 0 or it == 0:
            print(f"  Iter {it+1:>4d}/{n_iter}, err = {errors[-1]:.4e}")

    return {
        'object': obj,
        'probe': probe,
        'errors': errors
    }
