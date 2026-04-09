import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

import nmrglue as ng

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def forward_operator(fid_full, nus_schedule):
    """
    NUS forward operator: apply sampling mask to FID.
    
    Uses nmrglue's processing pipeline for apodization of the
    direct dimension, then masks the indirect dimension according
    to the NUS schedule.
    
    Parameters
    ----------
    fid_full : np.ndarray
        Complete 2D FID (n_f1 × n_f2).
    nus_schedule : np.ndarray
        Boolean mask of sampled t1 points.
    
    Returns
    -------
    fid_nus : np.ndarray
        NUS-sampled FID (same shape, zeros at unsampled t1 rows).
    """
    # Apply nmrglue apodization to F2 (direct dimension)
    fid_proc = ng.proc_base.em(fid_full, lb=5.0)

    # Apply NUS mask to indirect dimension (F1)
    fid_nus = np.zeros_like(fid_proc)
    fid_nus[nus_schedule, :] = fid_proc[nus_schedule, :]

    return fid_nus
