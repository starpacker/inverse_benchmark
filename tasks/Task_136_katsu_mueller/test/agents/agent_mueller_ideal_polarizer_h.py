import matplotlib

matplotlib.use('Agg')

import numpy as np

def mueller_ideal_polarizer_h():
    """Ideal horizontal linear polarizer."""
    return 0.5 * np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=float)
