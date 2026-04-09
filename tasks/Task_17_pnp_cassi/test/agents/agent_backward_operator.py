import numpy as np


# --- Extracted Dependencies ---

def backward_operator(y, Phi):
    """
    Transpose of the forward model.
    """
    return np.multiply(np.repeat(y[:, :, np.newaxis], Phi.shape[2], axis=2), Phi)
