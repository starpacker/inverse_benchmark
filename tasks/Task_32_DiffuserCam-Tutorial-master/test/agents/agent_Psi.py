import numpy as np

def Psi(v):
    # Forward finite difference
    return np.stack((np.roll(v, 1, axis=0) - v, np.roll(v, 1, axis=1) - v), axis=2)
