import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def phase_gradient_autofocus(phase_data, n_iter=5):
    """
    Phase Gradient Autofocus (PGA) for residual phase error correction.
    """
    data = phase_data.copy()
    n_pulses, n_range = data.shape

    for it in range(n_iter):
        # Range-compress
        rc_data = fftshift(fft(data, axis=1), axes=1)

        # For each range bin, estimate phase gradient
        phase_errors = np.zeros(n_pulses)
        for k in range(n_range):
            col = rc_data[:, k]
            # Shift to align peak
            peak_idx = np.argmax(np.abs(col))
            col_shifted = np.roll(col, n_pulses // 2 - peak_idx)

            # Phase gradient estimation
            if np.abs(col_shifted).max() > 1e-10:
                grad = np.angle(col_shifted[1:] * np.conj(col_shifted[:-1]))
                weight = np.abs(col_shifted[:-1])**2
                if weight.sum() > 1e-10:
                    phase_errors[:-1] += grad * weight
                    phase_errors[-1] = phase_errors[-2]

        # Normalise and integrate
        phase_errors /= max(np.abs(phase_errors).max(), 1e-10)
        correction = np.exp(-1j * np.cumsum(phase_errors))

        # Apply correction
        for n in range(n_pulses):
            data[n, :] *= correction[n]

    return data
