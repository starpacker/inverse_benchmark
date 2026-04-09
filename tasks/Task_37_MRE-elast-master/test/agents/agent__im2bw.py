import numpy as np

def _im2bw(Ig, level):
    S = np.copy(Ig)
    S[Ig > level] = 1
    S[Ig <= level] = 0
    return S
