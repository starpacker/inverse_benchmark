import matplotlib

matplotlib.use("Agg")

import numpy as np

from scipy import sparse

def _build_laplacian_2d(nr, nz):
    """Build a sparse 2D Laplacian operator for an (nr, nz) grid."""
    n = nr * nz
    diags = []
    offsets = []

    # Main diagonal: -4 (or fewer at boundaries handled by adjacency)
    main = -4.0 * np.ones(n)
    diags.append(main)
    offsets.append(0)

    # Right neighbour (+1 in z direction)
    d = np.ones(n - 1)
    # Zero out wrap-around at z boundaries
    for i in range(n - 1):
        if (i + 1) % nz == 0:
            d[i] = 0.0
    diags.append(d)
    offsets.append(1)

    # Left neighbour (-1 in z direction)
    d = np.ones(n - 1)
    for i in range(n - 1):
        if (i + 1) % nz == 0:
            d[i] = 0.0
    diags.append(d)
    offsets.append(-1)

    # Down neighbour (+nz in r direction)
    diags.append(np.ones(n - nz))
    offsets.append(nz)

    # Up neighbour (-nz in r direction)
    diags.append(np.ones(n - nz))
    offsets.append(-nz)

    Lap = sparse.diags(diags, offsets, shape=(n, n), format="csr")
    return Lap
