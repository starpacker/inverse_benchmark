import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift

from scipy.signal import windows

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

C = 3e8

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

def run_inversion(phase_noisy, u_positions, t_range, params):
    """
    Run SAR image reconstruction using backprojection and PFA methods.
    
    Implements:
    1) Backprojection (time-domain) image formation with matched-filter
    2) Polar-Format Algorithm (PFA) with range-compressed data
    3) Autofocus (Phase Gradient Autofocus) for phase error correction
    
    Args:
        phase_noisy: noisy phase history data
        u_positions: aperture positions
        t_range: range time samples
        params: dictionary of SAR parameters
        
    Returns:
        dict containing:
            - img_bp: backprojection reconstruction
            - img_pfa: PFA reconstruction
            - img_rec: best reconstruction (after post-processing)
            - method: name of best method used
    """
    n_pulses = params['n_pulses']
    n_range = params['n_range']
    scene_size = params['scene_size']
    r0 = params['r0']
    fc = params['fc']
    bandwidth = params['bandwidth']
    
    # ─── Backprojection Image Formation ───
    print("\n[STAGE 3a] Inverse — Backprojection Image Formation")
    
    nx, ny = n_pulses, n_range
    x_img = np.linspace(-scene_size / 2, scene_size / 2, nx)
    y_img = np.linspace(0, scene_size, ny)

    n_pulses_data = len(u_positions)
    image_bp = np.zeros((nx, ny), dtype=complex)

    print(f"  Matched-filter backprojecting {n_pulses_data} pulses onto {nx}×{ny} grid ...")

    chunk_size = 8  # process rows in chunks to manage memory
    for n in range(n_pulses_data):
        for ci in range(0, nx, chunk_size):
            ce = min(ci + chunk_size, nx)
            # Compute range from aperture position n to all pixels in this chunk
            xx_chunk = x_img[ci:ce, np.newaxis]  # (chunk, 1)
            yy_all = y_img[np.newaxis, :]         # (1, ny)
            R = np.sqrt((u_positions[n] - xx_chunk)**2 + r0**2 + yy_all**2)  # (chunk, ny)
            tau = 2 * R / C - 2 * r0 / C  # delay relative to reference (chunk, ny)
            phase = -4 * np.pi * fc * R / C  # (chunk, ny)

            # Sinc correlation for range focusing:
            t_diff = t_range[np.newaxis, np.newaxis, :] - tau[:, :, np.newaxis]
            sinc_vals = np.sinc(bandwidth * t_diff)  # (chunk, ny, n_range)

            # Inner product with data
            ip = np.sum(phase_noisy[n, :][np.newaxis, np.newaxis, :] * sinc_vals,
                        axis=2)  # (chunk, ny)

            # Phase compensation (matched filter)
            image_bp[ci:ce, :] += ip * np.exp(-1j * phase)

    img_bp = np.abs(image_bp)
    
    # ─── Polar Format Algorithm (PFA) ───
    print("\n[STAGE 3b] Inverse — Polar Format Algorithm")
    
    # Apply autofocus first
    phase_af = phase_gradient_autofocus(phase_noisy, n_iter=3)
    
    # PFA reconstruction
    n_pulses_pfa, n_range_pfa = phase_af.shape

    # Apply window
    range_window = windows.hamming(n_range_pfa)
    azimuth_window = windows.hamming(n_pulses_pfa)

    data_windowed = phase_af.copy()
    for n in range(n_pulses_pfa):
        data_windowed[n, :] *= range_window
    for k in range(n_range_pfa):
        data_windowed[:, k] *= azimuth_window

    # 2D FFT
    image_2d = fftshift(fft2(data_windowed, s=[nx, ny]))
    img_pfa = np.abs(image_2d)
    
    return {
        'img_bp': img_bp,
        'img_pfa': img_pfa
    }
