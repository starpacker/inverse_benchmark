import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

C = 3e8

def generate_sar_phase_history(sigma, n_pulses, n_range, aperture_length,
                                r0, fc, bandwidth, scene_size, rng, noise_snr_db):
    """
    Generate raw SAR phase-history data using the stripmap SAR model (vectorised).
    """
    u = np.linspace(-aperture_length / 2, aperture_length / 2, n_pulses)
    x_scene = np.linspace(-scene_size / 2, scene_size / 2, sigma.shape[0])
    y_scene = np.linspace(0, scene_size, sigma.shape[1])  # one-sided range (no y-ambiguity)
    range_extent = C / (2 * bandwidth) * n_range
    t_range = np.linspace(-range_extent / (2 * C), range_extent / (2 * C), n_range)

    phase_data = np.zeros((n_pulses, n_range), dtype=complex)
    print(f"  Generating phase history ({n_pulses} pulses × {n_range} range bins) ...")

    # Get non-zero target positions
    nz_idx = np.argwhere(sigma > 1e-10)
    if len(nz_idx) == 0:
        return phase_data, phase_data.copy(), u, t_range
    sigma_nz = sigma[nz_idx[:, 0], nz_idx[:, 1]]  # (K,)
    x_nz = x_scene[nz_idx[:, 0]]  # (K,)
    y_nz = y_scene[nz_idx[:, 1]]  # (K,)

    for n in range(n_pulses):
        # Vectorised over all targets
        R = np.sqrt((u[n] - x_nz)**2 + r0**2 + y_nz**2)  # (K,)
        tau = 2 * R / C
        # t_diff: (K, n_range) via broadcasting
        t_diff = t_range[np.newaxis, :] - (tau[:, np.newaxis] - 2 * r0 / C)
        envelope = np.sinc(bandwidth * t_diff)  # (K, n_range)
        phase = -4 * np.pi * fc * R / C  # (K,)
        contributions = sigma_nz[:, np.newaxis] * envelope * np.exp(1j * phase[:, np.newaxis])
        phase_data[n, :] = contributions.sum(axis=0)

    signal_power = np.mean(np.abs(phase_data)**2)
    noise_power = signal_power / (10**(noise_snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (rng.standard_normal(phase_data.shape) +
                                          1j * rng.standard_normal(phase_data.shape))
    phase_data_noisy = phase_data + noise
    return phase_data, phase_data_noisy, u, t_range
