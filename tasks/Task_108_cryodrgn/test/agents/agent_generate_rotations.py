import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.spatial.transform import Rotation

def generate_rotations(n_proj):
    """Generate random rotation matrices for projections."""
    return Rotation.random(n_proj, random_state=42).as_matrix().astype(np.float64)
