def normalize(x):
    """Normalize array to [0, 1] range."""
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-15)
