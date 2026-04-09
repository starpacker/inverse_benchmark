import numpy as np

import matplotlib

matplotlib.use('Agg')

def create_spiral_array(n_mics, radius):
    """Archimedean spiral microphone array in z=0 plane."""
    angles = np.linspace(0, 4 * np.pi, n_mics, endpoint=False)
    radii = np.linspace(0.05, radius, n_mics)
    return np.column_stack([radii * np.cos(angles),
                            radii * np.sin(angles),
                            np.zeros(n_mics)])
