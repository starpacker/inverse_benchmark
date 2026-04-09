import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def scale(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    out = v / norm
    return out * (1/np.max(np.abs(out)))
