import numpy as np


# --- Extracted Dependencies ---

def shift_back(inputs, step):
    """
    Reverse the dispersion effect.
    """
    row, col, nC = inputs.shape
    inputs_copy = inputs.copy()
    for i in range(nC):
        inputs_copy[:, :, i] = np.roll(inputs_copy[:, :, i], (-1) * step * i, axis=1)
    output = inputs_copy[:, 0:col - step * (nC - 1), :]
    return output
