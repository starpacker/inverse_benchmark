import numpy as np

from scipy.interpolate import interp1d

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

def load_and_preprocess_data(seed: int, fs: float, t_start: float, t_end: float,
                              f0: float, n_days: int, dvv_amp: float,
                              dvv_noise: float, dvv_period: float,
                              noise_level: float) -> dict:
    """
    Synthesize reference CCF and multi-day dataset with known dv/v.
    
    Returns a dictionary containing:
        - t: time axis array
        - ccf_ref: reference cross-correlation function
        - days: array of day indices
        - dvv_true: true dv/v values for each day
        - ccf_matrix: matrix of perturbed CCFs (n_days x n_samples)
    """
    rng = np.random.default_rng(seed)
    
    # Time axis
    n_samples = int((t_end - t_start) * fs) + 1
    t = np.linspace(t_start, t_end, n_samples)
    
    # Synthesise reference CCF
    ccf_ref = synthesise_reference_ccf(t, f0, rng)
    
    # Create synthetic dv/v time series
    days = np.arange(n_days)
    dvv_true = dvv_amp * np.sin(2.0 * np.pi * days / dvv_period) + \
        dvv_noise * rng.standard_normal(n_days)
    
    # Generate perturbed CCFs using forward operator
    ccf_matrix = np.empty((n_days, len(t)))
    for d in range(n_days):
        eps = -dvv_true[d]
        t_stretched = t * (1.0 + eps)
        interp_func = interp1d(t, ccf_ref, kind='cubic',
                               bounds_error=False, fill_value=0.0)
        ccf_cur = interp_func(t_stretched)
        ccf_cur += noise_level * rng.standard_normal(len(t)) * np.max(np.abs(ccf_ref))
        ccf_matrix[d] = ccf_cur
    
    return {
        "t": t,
        "ccf_ref": ccf_ref,
        "days": days,
        "dvv_true": dvv_true,
        "ccf_matrix": ccf_matrix,
    }
