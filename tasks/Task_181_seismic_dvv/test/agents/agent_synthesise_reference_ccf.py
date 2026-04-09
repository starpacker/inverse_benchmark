import numpy as np

def make_ricker(t: np.ndarray, f0: float) -> np.ndarray:
    """Ricker (Mexican-hat) wavelet centred at t=0."""
    a = 1.0 / f0
    u = (np.pi * (t) / a) ** 2
    return (1.0 - 2.0 * u) * np.exp(-u)

def synthesise_reference_ccf(t: np.ndarray, f0: float, rng,
                              decay_tau: float = 5.0,
                              n_scatterers: int = 12) -> np.ndarray:
    """
    Build a realistic-looking reference cross-correlation function.
    Ricker wavelet modulated by exponential decay + scattered arrivals.
    """
    ccf = make_ricker(t, f0) * np.exp(-np.abs(t) / decay_tau)
    for _ in range(n_scatterers):
        t_shift = rng.uniform(-8, 8)
        amp = rng.uniform(0.05, 0.25) * np.exp(-np.abs(t_shift) / decay_tau)
        f_scat = rng.uniform(1.5, 3.5)
        ccf += amp * make_ricker(t - t_shift, f_scat)
    return ccf
