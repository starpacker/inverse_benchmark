import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_model_visibilities(
    fluxes: np.ndarray,
    lm: np.ndarray,
    uvw: np.ndarray,
    freqs: np.ndarray,
    n_time: int,
) -> np.ndarray:
    """
    Compute model visibilities using the RIME for point sources.
    Returns: (n_bl, n_freq, n_time) complex array
    """
    c = 2.998e8
    n_bl = uvw.shape[0]
    n_freq = freqs.shape[0]
    n_src = fluxes.shape[0]
    freqs_hz = freqs * 1e9
    hour_angles = np.linspace(-0.5, 0.5, n_time)
    omega = 2.0 * np.pi / (24.0 * 3600.0)

    v_model = np.zeros((n_bl, n_freq, n_time), dtype=np.complex128)

    for t_idx, ha in enumerate(hour_angles):
        rot_angle = omega * ha * 3600.0
        cos_r, sin_r = np.cos(rot_angle), np.sin(rot_angle)
        u_rot = uvw[:, 0] * cos_r - uvw[:, 1] * sin_r
        v_rot = uvw[:, 0] * sin_r + uvw[:, 1] * cos_r

        for s_idx in range(n_src):
            l_s, m_s = lm[s_idx]
            ul_vm = u_rot * l_s + v_rot * m_s
            for f_idx in range(n_freq):
                phase = -2.0 * np.pi * ul_vm * freqs_hz[f_idx] / c
                v_model[:, f_idx, t_idx] += fluxes[s_idx, f_idx] * np.exp(1j * phase)

    return v_model
