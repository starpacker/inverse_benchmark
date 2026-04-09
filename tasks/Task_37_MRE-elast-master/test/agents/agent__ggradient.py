import numpy as np

def _ggradient(E, A):
    N = np.size(E)
    grad1 = np.zeros((N, 1))
    for n in range(N):
        S1 = 0
        for k in range(10):
            if A[n, k] != 0:
                neighbor_idx = int(A[n, k])
                if neighbor_idx < N:
                    S1 = S1 + (E[n] - E[neighbor_idx])**2
        grad1[n] = np.sqrt(S1)
    if np.max(grad1) != 0:
        result1 = grad1 / np.max(np.ceil(grad1))
    else:
        result1 = grad1
    return np.ravel(result1)
