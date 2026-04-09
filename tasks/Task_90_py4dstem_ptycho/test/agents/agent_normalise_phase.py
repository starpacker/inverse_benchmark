import matplotlib

matplotlib.use("Agg")

def normalise_phase(phase):
    """Shift to min=0 and normalise to [0, 1]."""
    p = phase - phase.min()
    mx = p.max()
    return p / mx if mx > 0 else p
