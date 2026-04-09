import numpy as np


# --- Extracted Dependencies ---

def Low_frequency_resolve(coeffs, dlevel):
    cAn = coeffs[0]
    vec = []
    vec.append(cAn)
    for i in range(1, dlevel + 1):
        (cH, cV, cD) = coeffs[i]
        [cH_x, cH_y] = cH.shape
        cH_new = np.zeros((cH_x, cH_y))
        t = (cH_new, cH_new, cH_new)
        vec.append(t)
    return vec
