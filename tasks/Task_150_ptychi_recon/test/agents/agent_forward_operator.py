import matplotlib

matplotlib.use('Agg')

import os

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def forward_operator(obj, probe, positions):
    """
    Compute diffraction intensities for ptychography.

    Forward model for each scan position j:
        exit_wave_j = probe * object_patch_j
        I_j = |FFT(exit_wave_j)|^2

    Parameters
    ----------
    obj : ndarray (complex)
        Complex-valued object transmission function.
    probe : ndarray (complex)
        Complex-valued probe illumination function.
    positions : list of tuples
        List of (py, px) scan positions.

    Returns
    -------
    list of ndarray
        List of diffraction intensity patterns for each scan position.
    """
    ph, pw = probe.shape
    intensities = []
    for py, px in positions:
        # Extract object patch at this scan position
        obj_patch = obj[py:py+ph, px:px+pw]
        # Compute exit wave
        exit_wave = probe * obj_patch
        # Fourier transform to get far-field diffraction
        fourier_wave = np.fft.fft2(exit_wave)
        # Compute intensity (squared modulus)
        intensity = np.abs(fourier_wave)**2
        intensities.append(intensity)
    return intensities
