import numpy as np

from scipy.signal import argrelmin, argrelmax

import brighteyes_ism.simulation.PSF_sim as psf_sim

def find_max_discrepancy(correlation: np.ndarray, gridpar: psf_sim.GridParameters, mode: str, graph: bool):
    if mode == 'KL':
        idx = np.asarray(argrelmax(correlation)).ravel()[0]
    elif mode == 'Pearson':
        idx = np.asarray(argrelmin(correlation)).ravel()[0]
    else:
        raise Exception("Discrepancy method unknown.")

    optimal_depth = idx * gridpar.pxsizez
    return optimal_depth
