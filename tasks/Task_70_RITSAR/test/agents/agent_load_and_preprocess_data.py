import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

C = 3e8

def generate_scene(n_range, n_cross, scene_size):
    """
    Create a scene with point targets + extended targets.
    Returns 2D reflectivity map σ(x,y).
    """
    sigma = np.zeros((n_cross, n_range))
    cx, cy = n_cross // 2, n_range // 2

    # Point targets at various positions
    targets = [
        (cx, cy, 1.0),           # Centre
        (cx - 15, cy - 20, 0.8),
        (cx + 10, cy + 15, 0.6),
        (cx - 20, cy + 25, 0.7),
        (cx + 25, cy - 10, 0.9),
        (cx + 5, cy + 30, 0.5),
    ]
    for tx, ty, amp in targets:
        if 0 <= tx < n_cross and 0 <= ty < n_range:
            sigma[tx, ty] = amp

    # Extended target: small rectangular structure
    sigma[cx-3:cx+3, cy+8:cy+12] = 0.4

    # L-shaped structure
    sigma[cx+10:cx+15, cy-15:cy-10] = 0.5
    sigma[cx+10:cx+12, cy-15:cy-5] = 0.5

    return sigma

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

def load_and_preprocess_data(n_pulses, n_range, aperture_length, r0, fc, bandwidth,
                              scene_size, noise_snr_db, seed):
    """
    Load and preprocess SAR data.
    Generates scene and phase history data for SAR image formation.
    
    Returns:
        dict containing:
            - sigma_gt: ground truth scene reflectivity
            - phase_clean: clean phase history data
            - phase_noisy: noisy phase history data
            - u_pos: aperture positions
            - t_range: range time samples
            - params: dictionary of SAR parameters
    """
    print("\n[STAGE 1] Data Generation")
    
    rng = np.random.default_rng(seed)
    
    # Generate scene
    sigma_gt = generate_scene(n_range, n_pulses, scene_size)
    print(f"  Scene: {sigma_gt.shape}, targets: {np.count_nonzero(sigma_gt)}")
    
    # Generate phase history
    phase_clean, phase_noisy, u_pos, t_range = generate_sar_phase_history(
        sigma_gt, n_pulses, n_range, aperture_length, r0, fc, bandwidth,
        scene_size, rng, noise_snr_db
    )
    print(f"  Phase history: {phase_noisy.shape}")
    print(f"  SNR: {noise_snr_db} dB")
    
    # Store parameters
    params = {
        'n_pulses': n_pulses,
        'n_range': n_range,
        'aperture_length': aperture_length,
        'r0': r0,
        'fc': fc,
        'bandwidth': bandwidth,
        'scene_size': scene_size,
        'noise_snr_db': noise_snr_db
    }
    
    return {
        'sigma_gt': sigma_gt,
        'phase_clean': phase_clean,
        'phase_noisy': phase_noisy,
        'u_pos': u_pos,
        't_range': t_range,
        'params': params
    }
