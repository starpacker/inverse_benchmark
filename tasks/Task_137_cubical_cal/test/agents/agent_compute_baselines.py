import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

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
