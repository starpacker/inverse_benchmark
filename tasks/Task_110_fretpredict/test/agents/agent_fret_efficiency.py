import matplotlib

matplotlib.use("Agg")

def fret_efficiency(r, R0):
    """E(r) = 1 / (1 + (r/R0)^6)."""
    return 1.0 / (1.0 + (r / R0) ** 6)
