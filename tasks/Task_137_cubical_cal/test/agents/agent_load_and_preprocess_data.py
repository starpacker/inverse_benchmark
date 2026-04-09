import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.ndimage import uniform_filter

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_sky_model(n_src: int, n_freq: int, rng: np.random.Generator) -> tuple:
    """
    Generate a simple point-source sky model.
    Returns source fluxes (n_src, n_freq) and direction cosines (n_src, 2).
    """
    s0 = rng.uniform(1.0, 10.0, size=n_src)
    alpha = rng.uniform(-1.5, -0.5, size=n_src)
    freqs = np.linspace(0.9, 1.7, n_freq)
    f0 = 1.3
    fluxes = s0[:, None] * (freqs[None, :] / f0) ** alpha[:, None]
    lm = rng.uniform(-0.01, 0.01, size=(n_src, 2))
    return fluxes, lm, freqs

def generate_antenna_layout(n_ant: int, rng: np.random.Generator) -> np.ndarray:
    """Generate antenna positions in metres (East-North-Up)."""
    positions = rng.uniform(-500.0, 500.0, size=(n_ant, 3))
    positions[:, 2] = 0.0
    return positions

def compute_baselines(positions: np.ndarray) -> tuple:
    """Compute baseline vectors and antenna-pair indices."""
    n_ant = positions.shape[0]
    ant1, ant2 = [], []
    for i in range(n_ant):
        for j in range(i + 1, n_ant):
            ant1.append(i)
            ant2.append(j)
    ant1 = np.array(ant1)
    ant2 = np.array(ant2)
    uvw = positions[ant2] - positions[ant1]
    return ant1, ant2, uvw

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

def generate_true_gains(n_ant: int, n_freq: int, n_time: int, ref_ant: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate true complex gains per antenna, per frequency, per time.
    Shape: (n_ant, n_freq, n_time)
    """
    amplitudes = np.zeros((n_ant, n_freq, n_time))
    phases_deg = np.zeros((n_ant, n_freq, n_time))

    for a in range(n_ant):
        amp_base = rng.uniform(0.8, 1.2)
        phase_base = rng.uniform(-30.0, 30.0)
        amp_var = rng.normal(0, 0.03, size=(n_freq, n_time))
        phase_var = rng.normal(0, 3.0, size=(n_freq, n_time))
        amp_var = uniform_filter(amp_var, size=3)
        phase_var = uniform_filter(phase_var, size=3)
        amplitudes[a] = amp_base + amp_var
        phases_deg[a] = phase_base + phase_var

    amplitudes[ref_ant] = 1.0
    phases_deg[ref_ant] = 0.0

    phases_rad = np.deg2rad(phases_deg)
    gains = amplitudes * np.exp(1j * phases_rad)
    return gains

def load_and_preprocess_data(
    n_ant: int,
    n_freq: int,
    n_time: int,
    n_src: int,
    snr_db: float,
    ref_ant: int,
    seed: int,
) -> dict:
    """
    Generate all synthetic data for radio interferometry calibration:
    - Sky model (point sources)
    - Antenna layout and baselines
    - Model visibilities
    - True gains
    - Observed (corrupted) visibilities
    
    Returns a dictionary containing all necessary data for calibration.
    """
    rng = np.random.default_rng(seed)
    
    # Generate sky model
    fluxes, lm, freqs = generate_sky_model(n_src, n_freq, rng)
    
    # Generate antenna layout and baselines
    positions = generate_antenna_layout(n_ant, rng)
    ant1, ant2, uvw = compute_baselines(positions)
    
    # Compute model visibilities
    v_model = compute_model_visibilities(fluxes, lm, uvw, freqs, n_time)
    
    # Generate true gains
    g_true = generate_true_gains(n_ant, n_freq, n_time, ref_ant, rng)
    
    # Store all data in a dictionary
    data = {
        'fluxes': fluxes,
        'lm': lm,
        'freqs': freqs,
        'positions': positions,
        'ant1': ant1,
        'ant2': ant2,
        'uvw': uvw,
        'v_model': v_model,
        'g_true': g_true,
        'n_ant': n_ant,
        'n_freq': n_freq,
        'n_time': n_time,
        'n_src': n_src,
        'snr_db': snr_db,
        'ref_ant': ref_ant,
        'rng': rng,
    }
    
    return data
