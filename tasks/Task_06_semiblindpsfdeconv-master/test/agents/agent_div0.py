import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def div0(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~ np.isfinite(c)] = 0
    return c
