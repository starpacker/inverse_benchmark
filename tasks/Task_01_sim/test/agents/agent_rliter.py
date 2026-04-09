import numpy as np


# --- Extracted Dependencies ---

def rliter(yk, data, otf):
    rliter_val = np.fft.fftn(data / np.maximum(np.fft.ifftn(otf * np.fft.fftn(yk)), 1e-6))
    return rliter_val
