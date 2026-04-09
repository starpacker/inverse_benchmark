import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def arange2(start, stop=None, step=1):
    """Modified version of numpy.arange which corrects error associated with non-integer step size"""
    if stop is None:
        a = np.arange(start)
    else:
        a = np.arange(start, stop, step)
        if a[-1] > stop - step:
            a = np.delete(a, -1)
    return a
