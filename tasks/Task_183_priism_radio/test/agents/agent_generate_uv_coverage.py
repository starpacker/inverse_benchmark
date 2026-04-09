import numpy as np

import matplotlib

matplotlib.use('Agg')

def generate_uv_coverage(n_antennas=10, n_hours=6, n_time_steps=60, rng=None):
    """
    Simulate (u,v) coverage from an interferometric array via
    earth-rotation synthesis.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    max_baseline = 50.0
    ant_E = rng.uniform(-max_baseline, max_baseline, n_antennas)
    ant_N = rng.uniform(-max_baseline, max_baseline, n_antennas)

    dec = np.deg2rad(45.0)
    ha = np.linspace(-n_hours / 2, n_hours / 2, n_time_steps) * (np.pi / 12)

    u_all, v_all = [], []
    for i in range(n_antennas):
        for j in range(i + 1, n_antennas):
            bE = ant_E[j] - ant_E[i]
            bN = ant_N[j] - ant_N[i]
            u_t = bE * np.cos(ha) - bN * np.sin(ha) * np.sin(dec)
            v_t = bE * np.sin(ha) * np.sin(dec) + bN * np.cos(ha) * np.cos(dec)
            u_all.append(u_t)
            v_all.append(v_t)

    u = np.concatenate(u_all)
    v = np.concatenate(v_all)

    # Include conjugate baselines (Hermitian symmetry)
    u = np.concatenate([u, -u])
    v = np.concatenate([v, -v])

    return u, v
