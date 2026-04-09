import numpy as np

def inverse_tikhonov(matrix, rcond=1e-3):
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    S_inv = S / (S**2 + rcond**2 * S.max()**2)
    return Vt.T.dot(np.diag(S_inv)).dot(U.T)
