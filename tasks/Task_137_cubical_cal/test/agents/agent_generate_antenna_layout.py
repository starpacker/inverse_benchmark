import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_antenna_layout(n_ant: int, rng: np.random.Generator) -> np.ndarray:
    """Generate antenna positions in metres (East-North-Up)."""
    positions = rng.uniform(-500.0, 500.0, size=(n_ant, 3))
    positions[:, 2] = 0.0
    return positions
