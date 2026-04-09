import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

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
