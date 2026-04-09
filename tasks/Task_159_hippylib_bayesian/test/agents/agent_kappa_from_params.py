import numpy as np

import matplotlib

matplotlib.use('Agg')

def kappa_from_params(params, X, Y, centers, sigma_basis, kappa_bg=1.0):
    """
    kappa(x,y) = kappa_bg + sum_i params[i] * G_i(x,y)
    where G_i is a Gaussian centered at centers[i].
    """
    kappa = np.full_like(X, kappa_bg)
    for i in range(len(params)):
        cx, cy = centers[i]
        kappa += params[i] * np.exp(
            -((X - cx)**2 + (Y - cy)**2) / (2.0 * sigma_basis**2)
        )
    return np.maximum(kappa, 0.1)
