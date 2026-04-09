import matplotlib

matplotlib.use('Agg')

def harvey_comp(freq, zeta, nc):
    """Single Harvey component: P(ν) = ζ / (1 + (ν/ν_c)²)"""
    return zeta / (1.0 + (freq / nc) ** 2)
