import matplotlib

matplotlib.use("Agg")

def relative_error(true_val, est_val):
    """Compute relative error between true and estimated values."""
    return abs(est_val - true_val) / abs(true_val) if true_val != 0 else 0.0
