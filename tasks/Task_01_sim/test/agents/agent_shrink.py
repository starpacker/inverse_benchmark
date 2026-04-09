import numpy as np


# --- Extracted Dependencies ---

def shrink(x, L):
    s = np.abs(x)
    xs = np.sign(x) * np.maximum(s - 1 / L, 0)
    return xs
