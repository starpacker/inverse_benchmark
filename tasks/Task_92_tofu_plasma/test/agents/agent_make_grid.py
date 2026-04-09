import matplotlib

matplotlib.use("Agg")

import numpy as np

def make_grid(nr, nz, r_min, r_max, z_min, z_max):
    """Return 1-D R, Z arrays and the 2-D meshgrid."""
    r = np.linspace(r_min, r_max, nr)
    z = np.linspace(z_min, z_max, nz)
    RR, ZZ = np.meshgrid(r, z, indexing="ij")  # shape (NR, NZ)
    return r, z, RR, ZZ
