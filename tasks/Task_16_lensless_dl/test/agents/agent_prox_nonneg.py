import numpy as np


# --- Extracted Dependencies ---

def prox_nonneg(x):
    """Proximal operator for non-negativity constraint."""
    return np.maximum(x, 0)
