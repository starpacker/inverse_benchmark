import numpy as np

import matplotlib

matplotlib.use('Agg')

def williams_mode2_ux(r, theta, n, mu, kappa):
    """Mode II contribution to u_x for term index n."""
    rn = r ** (n / 2.0)
    val = (kappa + n / 2.0 - ((-1) ** n)) * np.sin(n * theta / 2.0) \
          - (n / 2.0) * np.sin((n / 2.0 - 1.0) * theta)
    return rn / (2.0 * mu) * val
