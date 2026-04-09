import logging

import numpy as np

from functools import reduce

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def divergence(F):
    return reduce(np.add, np.gradient(F))
