import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

C = 3e8

def forward_operator(sigma, u_positions, t_range, r0, fc, scene_size, bandwidth):
    """
    Forward SAR model: scene → phase history (vectorised).
    
    SAR phase-history model:
    s(u,t) = ∫∫ σ(x,y) · exp(-j·4π/λ · R(u,x,y)) dx dy
    where R(u,x,y) is the range from aperture position u to scene point (x,y).
    
    Args:
        sigma: 2D scene reflectivity array
        u_positions: aperture positions array
        t_range: range time samples array
        r0: reference range
        fc: carrier frequency
        scene_size: extent of scene
        bandwidth: chirp bandwidth
        
    Returns:
        phase_data: complex phase history array (n_pulses, n_range)
    """
    n_pulses = len(u_positions)
    n_range = len(t_range)
    x_scene = np.linspace(-scene_size / 2, scene_size / 2, sigma.shape[0])
    y_scene = np.linspace(0, scene_size, sigma.shape[1])  # one-sided range

    nz_idx = np.argwhere(sigma > 1e-10)
    if len(nz_idx) == 0:
        return np.zeros((n_pulses, n_range), dtype=complex)
    sigma_nz = sigma[nz_idx[:, 0], nz_idx[:, 1]]
    x_nz = x_scene[nz_idx[:, 0]]
    y_nz = y_scene[nz_idx[:, 1]]

    phase_data = np.zeros((n_pulses, n_range), dtype=complex)
    for n in range(n_pulses):
        R = np.sqrt((u_positions[n] - x_nz)**2 + r0**2 + y_nz**2)
        tau = 2 * R / C
        t_diff = t_range[np.newaxis, :] - (tau[:, np.newaxis] - 2 * r0 / C)
        envelope = np.sinc(bandwidth * t_diff)
        phase = -4 * np.pi * fc * R / C
        contributions = sigma_nz[:, np.newaxis] * envelope * np.exp(1j * phase[:, np.newaxis])
        phase_data[n, :] = contributions.sum(axis=0)
    return phase_data
