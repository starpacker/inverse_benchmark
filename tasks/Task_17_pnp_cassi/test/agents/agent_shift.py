import numpy as np


# --- Extracted Dependencies ---

def shift(inputs, step):
    """
    Simulate the dispersion effect (spatial shift).
    """
    row, col, nC = inputs.shape
    output = np.zeros((row, col + (nC - 1) * step, nC))
    for i in range(nC):
        output[:, i * step:i * step + col, i] = inputs[:, :, i]
    return output
