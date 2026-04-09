import numpy as np


# --- Extracted Dependencies ---

def shrink(x, L):
    s = np.abs(x)
    xs = np.sign(x) * np.maximum(s - 1 / L, 0)
    return xs

def iter_sparse(gsparse, bsparse, para, mu):
    dsparse = shrink(gsparse + bsparse, mu)
    bsparse = bsparse + (gsparse - dsparse)
    Lsparse = para * (dsparse - bsparse)
    return Lsparse, bsparse
