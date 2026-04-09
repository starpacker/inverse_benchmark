import numpy as np

def make_ricker(t: np.ndarray, f0: float) -> np.ndarray:
    """Ricker (Mexican-hat) wavelet centred at t=0."""
    a = 1.0 / f0
    u = (np.pi * (t) / a) ** 2
    return (1.0 - 2.0 * u) * np.exp(-u)
