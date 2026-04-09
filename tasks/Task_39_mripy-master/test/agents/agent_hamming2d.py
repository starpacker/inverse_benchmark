import numpy as np

import matplotlib

matplotlib.use('Agg')

def hamming2d(a, b):
    w2d = np.outer(np.hamming(a), np.hamming(b))
    return np.sqrt(w2d)
