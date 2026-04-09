import numpy as np

from scipy import sparse

import matplotlib

matplotlib.use('Agg')

def build_observation_operator(obs_iy, obs_ix, Ny, Nx):
    """Sparse observation matrix that picks values at sensor locations."""
    n_obs = len(obs_iy)
    N = Ny * Nx
    rows = np.arange(n_obs)
    cols = obs_iy * Nx + obs_ix
    return sparse.csr_matrix((np.ones(n_obs), (rows, cols)), shape=(n_obs, N))
