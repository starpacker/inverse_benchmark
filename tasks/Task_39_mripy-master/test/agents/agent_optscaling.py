import numpy as np

import matplotlib

matplotlib.use('Agg')

def optscaling(FT, b):
    x0 = np.absolute(FT.backward(b))
    return max(x0.flatten())
