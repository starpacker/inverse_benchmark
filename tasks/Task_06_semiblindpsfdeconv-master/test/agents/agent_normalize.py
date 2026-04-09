import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def normalize(v):
    norm = v.sum()
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v/norm
