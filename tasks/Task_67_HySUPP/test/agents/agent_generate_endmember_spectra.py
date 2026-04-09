import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_endmember_spectra(n_bands, n_end, rng):
    """
    Create synthetic endmember spectra resembling mineral reflectances.
    """
    wavelengths = np.linspace(400, 2500, n_bands)
    E = np.zeros((n_bands, n_end))

    specs = [
        (0.8, [(550, 40, 0.55), (1200, 60, 0.50)]),
        (0.15, [(900, 50, 0.10)]),
        (0.5, [(480, 40, 0.40), (2200, 100, 0.45)]),
        (0.35, [(700, 60, 0.20), (1600, 100, 0.30)]),
    ]

    for i in range(n_end):
        base, features = specs[i % len(specs)]
        E[:, i] = base
        for center, width, depth in features:
            E[:, i] -= depth * np.exp(-(wavelengths - center)**2 / (2 * width**2))
        E[:, i] += 0.02 * rng.standard_normal(n_bands)
        E[:, i] = np.clip(E[:, i], 0.01, 1.0)

    return E, wavelengths
