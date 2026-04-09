import matplotlib

matplotlib.use("Agg")

import numpy as np

def make_phantom(RR, ZZ):
    """
    Create a realistic tokamak-like emission phantom.
    Peaked profile centred at (R0, Z0) with an elliptical Gaussian shape
    plus a secondary weaker blob to add asymmetry.
    """
    R0, Z0 = 1.75, 0.0       # magnetic axis
    sigma_r, sigma_z = 0.30, 0.35

    # Main peaked profile
    eps = np.exp(-((RR - R0) ** 2 / (2 * sigma_r ** 2)
                   + (ZZ - Z0) ** 2 / (2 * sigma_z ** 2)))

    # Secondary blob (HFS accumulation)
    R1, Z1 = 1.45, 0.15
    sig1_r, sig1_z = 0.12, 0.10
    eps += 0.35 * np.exp(-((RR - R1) ** 2 / (2 * sig1_r ** 2)
                           + (ZZ - Z1) ** 2 / (2 * sig1_z ** 2)))

    # Clip outside last closed flux surface (rough ellipse)
    a_r, a_z = 0.60, 0.70
    mask = ((RR - R0) / a_r) ** 2 + ((ZZ - Z0) / a_z) ** 2 <= 1.0
    eps *= mask.astype(float)

    # Normalise to [0, 1]
    eps /= eps.max()
    return eps
