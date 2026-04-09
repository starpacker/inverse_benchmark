import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def forward_operator(E, A):
    """
    Linear spectral mixing model: Y = E @ A
    
    Args:
        E: Endmember matrix (L x R) - spectral signatures
        A: Abundance matrix (R x P) - fractional abundances
        
    Returns:
        Y_pred: Predicted mixed spectra (L x P)
    """
    Y_pred = E @ A
    return Y_pred
