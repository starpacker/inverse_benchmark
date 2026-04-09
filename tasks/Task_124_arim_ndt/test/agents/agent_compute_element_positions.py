import numpy as np

import matplotlib

matplotlib.use('Agg')

def compute_element_positions(n_elements, pitch):
    """Return 1-D array of element x-positions centered on 0."""
    return (np.arange(n_elements) - (n_elements - 1) / 2.0) * pitch
