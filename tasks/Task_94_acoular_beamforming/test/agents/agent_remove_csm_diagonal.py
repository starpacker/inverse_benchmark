import numpy as np

import matplotlib

matplotlib.use('Agg')

def remove_csm_diagonal(C):
    """Set CSM diagonal to zero to remove uncorrelated noise."""
    C_clean = C.copy()
    np.fill_diagonal(C_clean, 0)
    return C_clean
