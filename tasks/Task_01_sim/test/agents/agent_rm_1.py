import numpy as np


# --- Extracted Dependencies ---

def rm_1(Biter, x, y):
    Biter_new = np.zeros((x, y), dtype=('uint8'))
    if x % 2 and y % 2 == 0:
        Biter_new[:, :] = Biter[0:x, :]
    elif x % 2 == 0 and y % 2:
        Biter_new[:, :] = Biter[:, 0:y]
    elif x % 2 and y % 2:
        Biter_new[:, :] = Biter[0:x, 0:y]
    else:
        Biter_new = Biter
    return Biter_new
