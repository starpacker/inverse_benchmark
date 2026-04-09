import matplotlib

matplotlib.use("Agg")

import numpy as np

def load_and_preprocess_data(
    nx: int,
    ny: int,
    n_time: int,
    tau1_ns: float,
    tau2_ns: float,
    freq_mhz: float,
    total_photons: int,
    rng_seed: int,
) -> dict:
    """
    Generate synthetic FLIM data with ground truth fraction maps.
    
    Creates a 2D image with spatially varying fractions of two fluorophore
    species with known lifetimes. Generates time-resolved fluorescence decay
    at each pixel and adds Poisson noise.
    
    Parameters
    ----------
    nx, ny : int
        Image dimensions.
    n_time : int
        Number of time bins in the decay.
    tau1_ns, tau2_ns : float
        Lifetimes of species 1 and 2 in nanoseconds.
    freq_mhz : float
        Laser repetition frequency in MHz.
    total_photons : int
        Mean total photon count per pixel.
    rng_seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'f1_gt': Ground truth fraction map for species 1 (nx, ny)
        - 'f2_gt': Ground truth fraction map for species 2 (nx, ny)
        - 'noisy_signal': Noisy FLIM data (nx, ny, n_time)
        - 'clean_signal': Clean FLIM data (nx, ny, n_time)
        - 't': Time axis array (n_time,)
        - 'decay1': Normalized decay for species 1 (n_time,)
        - 'decay2': Normalized decay for species 2 (n_time,)
        - 'tau1_ns': Lifetime of species 1
        - 'tau2_ns': Lifetime of species 2
        - 'freq_mhz': Laser repetition frequency
    """
    rng = np.random.default_rng(rng_seed)
    
    # Create ground-truth fraction map (species-1 fraction varies spatially)
    yy, xx = np.meshgrid(np.linspace(0, 1, ny), np.linspace(0, 1, nx))
    
    # Base: horizontal gradient
    f1_gt = 0.2 + 0.6 * xx  # ranges from 0.2 to 0.8
    
    # Add a circular region of high species-2 (low f1)
    cx, cy, r = 0.65, 0.35, 0.18
    mask_circle = ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2
    f1_gt[mask_circle] = 0.15
    
    # Add a square region of high species-1
    f1_gt[20:45, 80:105] = 0.90
    
    f1_gt = np.clip(f1_gt, 0.0, 1.0).astype(np.float64)
    f2_gt = 1.0 - f1_gt
    
    print(f"Ground-truth fraction map: f1 range [{f1_gt.min():.3f}, {f1_gt.max():.3f}]")
    
    # Time axis (one laser period)
    period_ns = 1e3 / freq_mhz  # 12.5 ns for 80 MHz
    t = np.linspace(0, period_ns, n_time, endpoint=False)
    
    # Exponential decays for each species (normalized to integrate to 1)
    decay1 = np.exp(-t / tau1_ns)
    decay1 /= decay1.sum()
    
    decay2 = np.exp(-t / tau2_ns)
    decay2 /= decay2.sum()
    
    # Per-pixel decay = f1*decay1 + f2*decay2, scaled by total photon count
    clean_signal = (
        f1_gt[:, :, np.newaxis] * decay1[np.newaxis, np.newaxis, :]
        + f2_gt[:, :, np.newaxis] * decay2[np.newaxis, np.newaxis, :]
    ) * total_photons
    
    # Add Poisson noise
    noisy_signal = rng.poisson(np.maximum(clean_signal, 1e-12)).astype(np.float64)
    
    print(f"FLIM data shape: {noisy_signal.shape}, mean counts/pixel: {noisy_signal.sum(axis=-1).mean():.0f}")
    
    return {
        'f1_gt': f1_gt,
        'f2_gt': f2_gt,
        'noisy_signal': noisy_signal,
        'clean_signal': clean_signal,
        't': t,
        'decay1': decay1,
        'decay2': decay2,
        'tau1_ns': tau1_ns,
        'tau2_ns': tau2_ns,
        'freq_mhz': freq_mhz,
    }
