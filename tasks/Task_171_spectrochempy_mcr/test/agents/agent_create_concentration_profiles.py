import matplotlib

matplotlib.use('Agg')

import numpy as np

import os

np.random.seed(42)

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

def create_concentration_profiles(n_samples, n_components):
    """Create sinusoidal concentration profiles (simulating kinetics)."""
    t = np.linspace(0, 2 * np.pi, n_samples)
    C = np.zeros((n_samples, n_components))

    # Component 1: decaying sinusoidal
    C[:, 0] = 0.5 * (1 + np.sin(t)) * np.exp(-0.2 * t) + 0.1

    # Component 2: growing then decaying
    C[:, 1] = 0.8 * np.sin(t + np.pi / 3) ** 2 + 0.05

    # Component 3: delayed growth
    C[:, 2] = 0.6 * (1 - np.exp(-0.5 * t)) * np.abs(np.cos(t / 2)) + 0.1

    # Ensure non-negativity
    C = np.maximum(C, 0)
    return C
