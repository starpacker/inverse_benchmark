import matplotlib

matplotlib.use('Agg')

def harvey_comp(freq, zeta, nc):
    """Single Harvey component: P(ν) = ζ / (1 + (ν/ν_c)²)"""
    return zeta / (1.0 + (freq / nc) ** 2)

def bg_model(freq, z1, nc1, z2, nc2, w):
    """Background = 2 Harvey components + white noise."""
    return harvey_comp(freq, z1, nc1) + harvey_comp(freq, z2, nc2) + w
