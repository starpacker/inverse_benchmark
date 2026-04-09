import numpy as np

import matplotlib

matplotlib.use("Agg")

def sample_distances(n, r_max):
    """Sample n distances from the Gaussian mixture."""
    n1 = int(0.6 * n)
    n2 = n - n1
    d1 = np.random.normal(4.0, 0.5, n1)
    d2 = np.random.normal(7.0, 0.8, n2)
    distances = np.concatenate([d1, d2])
    np.random.shuffle(distances)
    distances = np.clip(distances, 0.01, r_max)
    return distances
