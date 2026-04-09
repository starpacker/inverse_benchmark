import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.fft import fft, ifft, fft2, ifft2, fftshift

import nmrglue as ng

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def soft_threshold(x, thresh):
    """Complex soft thresholding."""
    mag = np.abs(x)
    return np.where(mag > thresh, x * (1 - thresh / np.maximum(mag, 1e-30)), 0)

def run_inversion(fid_nus, schedule, ist_iterations, ist_threshold_decay):
    """
    Reconstruct 2D NMR spectrum from NUS data using IST
    (Iterative Soft Thresholding).
    
    Uses nmrglue for processing the direct dimension (apodization,
    FFT) and IST for the indirect dimension.
    
    Parameters
    ----------
    fid_nus : np.ndarray
        NUS-sampled FID.
    schedule : np.ndarray
        Boolean NUS sampling mask.
    ist_iterations : int
        Number of IST iterations.
    ist_threshold_decay : float
        Threshold decrease per iteration.
    
    Returns
    -------
    spec_recon : np.ndarray
        Reconstructed 2D spectrum.
    """
    print(f"[RECON] IST reconstruction ({ist_iterations} iterations) ...")

    # Process F2 (direct) dimension: apodization + FFT via nmrglue
    fid_f2 = ng.proc_base.em(fid_nus, lb=5.0)
    data_f2 = np.zeros_like(fid_f2, dtype=complex)
    for i in range(fid_f2.shape[0]):
        data_f2[i, :] = fft(fid_f2[i, :])

    # IST on F1 (indirect) dimension
    current = data_f2.copy()

    # Initial threshold from max of zero-filled spectrum
    zf_spec = fftshift(fft2(fid_f2)).real
    thresh = 0.99 * np.abs(zf_spec).max()

    for it in range(ist_iterations):
        # Step 1: FFT along F1
        spec = np.zeros_like(current, dtype=complex)
        for j in range(current.shape[1]):
            spec[:, j] = fft(current[:, j])

        # Step 2: Soft threshold in frequency domain
        spec_thresh = soft_threshold(spec, thresh)

        # Step 3: iFFT back to time domain
        for j in range(current.shape[1]):
            current[:, j] = ifft(spec_thresh[:, j])

        # Step 4: Enforce data consistency at sampled points
        current[schedule, :] = data_f2[schedule, :]

        # Decay threshold
        thresh *= ist_threshold_decay

        if (it + 1) % 50 == 0:
            residual = np.linalg.norm(current[schedule, :] - data_f2[schedule, :])
            print(f"[RECON]   iter {it+1:4d}  thresh={thresh:.4e}  "
                  f"residual={residual:.4e}")

    # Final spectrum
    spec_recon = np.zeros_like(current, dtype=complex)
    for j in range(current.shape[1]):
        spec_recon[:, j] = fft(current[:, j])

    spec_recon = fftshift(spec_recon).real
    spec_recon = spec_recon / np.abs(spec_recon).max()

    return spec_recon
