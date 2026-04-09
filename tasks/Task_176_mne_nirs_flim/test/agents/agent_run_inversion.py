import matplotlib

matplotlib.use('Agg')

import numpy as np

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

EPS_HBO_760 = 1486.5865

EPS_HBR_760 = 3843.707

EPS_HBO_850 = 2526.391

EPS_HBR_850 = 1798.643

DPF_760 = 6.0

DPF_850 = 5.5

D = 3.0

def run_inversion(od_760_noisy, od_850_noisy):
    """
    Inverse MBLL: solve 2×2 linear system at each time point.
    A · [HbO, HbR]^T = [ΔOD_760, ΔOD_850]^T
    
    Parameters
    ----------
    od_760_noisy : ndarray
        Noisy optical density at 760nm
    od_850_noisy : ndarray
        Noisy optical density at 850nm
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'hbo_rec': recovered HbO concentration
        - 'hbr_rec': recovered HbR concentration
    """
    A = np.array([
        [EPS_HBO_760 * DPF_760 * D, EPS_HBR_760 * DPF_760 * D],
        [EPS_HBO_850 * DPF_850 * D, EPS_HBR_850 * DPF_850 * D]
    ])
    
    # Stack observations: shape (2, N)
    od_stack = np.stack([od_760_noisy, od_850_noisy], axis=0)
    
    # Solve A x = b  →  x = A^{-1} b
    A_inv = np.linalg.inv(A)
    x = A_inv @ od_stack  # (2, N)
    
    hbo_rec = x[0]
    hbr_rec = x[1]
    
    result = {
        'hbo_rec': hbo_rec,
        'hbr_rec': hbr_rec
    }
    
    return result
