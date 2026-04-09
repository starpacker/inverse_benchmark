import numpy as np

def SoftThresh(x, tau):
    return np.sign(x) * np.maximum(0, np.abs(x) - tau)
