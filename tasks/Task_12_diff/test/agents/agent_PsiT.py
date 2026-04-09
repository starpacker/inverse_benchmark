import numpy as np

def PsiT(U):
    # Adjoint of forward finite difference
    diff1 = np.roll(U[..., 0], -1, axis=0) - U[..., 0]
    diff2 = np.roll(U[..., 1], -1, axis=1) - U[..., 1]
    return diff1 + diff2
