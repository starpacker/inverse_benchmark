import numpy as np

def minus_log(data):
    """
    Computes the minus log of the data: P = -log(data).
    """
    data = np.where(data <= 0, 1e-6, data)
    return -np.log(data)
