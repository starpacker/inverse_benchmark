import numpy as np

import matplotlib

matplotlib.use("Agg")

def true_distance_pdf(r):
    """Mixture of two Gaussians: w1·N(mu1,sig1²) + w2·N(mu2,sig2²)."""
    w1, mu1, sig1 = 0.6, 4.0, 0.5
    w2, mu2, sig2 = 0.4, 7.0, 0.8
    g1 = w1 / (sig1 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((r - mu1) / sig1) ** 2)
    g2 = w2 / (sig2 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((r - mu2) / sig2) ** 2)
    return g1 + g2
