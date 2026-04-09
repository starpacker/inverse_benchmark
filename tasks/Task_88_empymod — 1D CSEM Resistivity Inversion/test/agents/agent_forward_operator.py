import numpy as np

import matplotlib

matplotlib.use('Agg')

import warnings

warnings.filterwarnings('ignore')

def forward_operator(depth, res, frequencies, offsets, src_depth_abs, rec_depth_abs):
    """
    Compute CSEM frequency-domain EM response using empymod.
    
    Args:
        depth: List of layer interface depths
        res: List of layer resistivities
        frequencies: Array of frequencies (Hz)
        offsets: Array of source-receiver offsets (m)
        src_depth_abs: Absolute source depth (m)
        rec_depth_abs: Absolute receiver depth (m)
    
    Returns:
        response: Complex E-field array of shape (n_freq, n_off)
    """
    import empymod
    
    n_freq = len(frequencies)
    n_off = len(offsets)
    response = np.zeros((n_freq, n_off), dtype=complex)
    
    for j, offset in enumerate(offsets):
        resp = empymod.dipole(
            src=[0, 0, src_depth_abs],
            rec=[offset, 0, rec_depth_abs],
            depth=depth,
            res=res,
            freqtime=frequencies,
            verb=0,
        )
        response[:, j] = resp
    
    return response
