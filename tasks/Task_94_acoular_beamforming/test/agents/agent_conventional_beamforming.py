import numpy as np

import matplotlib

matplotlib.use('Agg')

def conventional_beamforming(C, G):
    """Vectorized delay-and-sum beamforming."""
    CG = C @ G
    B = np.real(np.sum(G.conj() * CG, axis=0))
    norms = np.real(np.sum(G.conj() * G, axis=0))
    norms_sq = np.maximum(norms**2, 1e-30)
    return np.maximum(B / norms_sq, 0)
