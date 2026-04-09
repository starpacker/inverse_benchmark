import numpy as np

import matplotlib

matplotlib.use('Agg')

SPECIES_BANDS = {
    "H2O": [
        (1.4e-6,  0.15e-6,  1.0e-25),
        (1.85e-6, 0.12e-6,  6.0e-26),
        (2.7e-6,  0.20e-6,  1.5e-25),
    ],
    "CH4": [
        (1.65e-6, 0.10e-6,  8.0e-26),
        (2.3e-6,  0.15e-6,  5.0e-26),
        (3.3e-6,  0.25e-6,  1.2e-25),
    ],
    "CO2": [
        (4.3e-6,  0.20e-6,  2.0e-25),
        (2.0e-6,  0.08e-6,  2.0e-26),
    ],
}

def compute_cross_section(wavelengths, species_name):
    """
    Compute simplified absorption cross-section for a species.
    Uses sum of Gaussian absorption bands.
    """
    bands = SPECIES_BANDS[species_name]
    sigma = np.zeros_like(wavelengths)
    for center, width, peak in bands:
        sigma += peak * np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
    return sigma
