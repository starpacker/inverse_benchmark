import numpy as np

import matplotlib

matplotlib.use('Agg')

def williams_mode1_ux(r, theta, n, mu, kappa):
    """Mode I contribution to u_x for term index n."""
    rn = r ** (n / 2.0)
    val = (kappa + n / 2.0 + ((-1) ** n)) * np.cos(n * theta / 2.0) \
          - (n / 2.0) * np.cos((n / 2.0 - 1.0) * theta)
    return rn / (2.0 * mu) * val
