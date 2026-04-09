import numpy as np

def _nodeneighbor(triangles, N):
    T = [[] for _ in range(N)]
    for elem in triangles:
        for node in elem:
            T[node].extend(elem)
    for i in range(N):
        T[i] = list(set(T[i]))
    T1 = np.zeros((N, 10))
    for i in range(N):
        length = min(len(T[i]), 10)
        T1[i, :length] = T[i][:length]
    return T1
