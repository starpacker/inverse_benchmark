import numpy as np

def compute_pad_size(width: int):
    return int(np.ceil((np.sqrt(2)*width - width)/2))
