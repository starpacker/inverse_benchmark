import numpy as np

import matplotlib

matplotlib.use('Agg')

def compute_steering_vectors(mic_positions, grid_points, k):
    """Steering vector matrix G[i,j] = exp(-jkr) / (4πr)."""
    diff = mic_positions[:, np.newaxis, :] - grid_points[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    return np.exp(-1j * k * distances) / (4.0 * np.pi * distances)
