import matplotlib

matplotlib.use("Agg")

import numpy as np

from scipy.signal import fftconvolve

def load_and_preprocess_data(
    n_channels: int,
    dt: float,
    irf_center: float,
    irf_fwhm: float,
    tau1_true: float,
    tau2_true: float,
    a1_true: float,
    a2_true: float,
    background: float,
    total_counts: int,
    rng_seed: int
) -> dict:
    """
    Generate synthetic TCSPC histogram data.
    
    Creates time axis, IRF, ground truth decay curve, and noisy measured data.
    
    Parameters
    ----------
    n_channels : int
        Number of time bins.
    dt : float
        Time step per channel (ns).
    irf_center : float
        Center of Gaussian IRF (ns).
    irf_fwhm : float
        Full width at half maximum of IRF (ns).
    tau1_true : float
        Short lifetime (ns).
    tau2_true : float
        Long lifetime (ns).
    a1_true : float
        Amplitude fraction for tau1.
    a2_true : float
        Amplitude fraction for tau2.
    background : float
        Constant background counts per bin.
    total_counts : int
        Total number of photon counts.
    rng_seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'time': time axis array (ns)
        - 'irf': normalized IRF array
        - 'gt_curve': ground truth curve without background
        - 'gt_curve_with_bg': ground truth curve with background
        - 'measured': noisy Poisson measurement
        - 'params': dict of true parameters
    """
    rng = np.random.default_rng(rng_seed)
    
    # Time axis
    time = np.arange(n_channels) * dt
    
    # IRF - Gaussian
    irf_sigma = irf_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    irf = np.exp(-0.5 * ((time - irf_center) / irf_sigma) ** 2)
    irf = irf / irf.sum()
    
    # Bi-exponential decay
    decay = a1_true * np.exp(-time / tau1_true) + a2_true * np.exp(-time / tau2_true)
    
    # Convolve IRF with decay
    convolved = fftconvolve(irf, decay, mode="full")[:len(time)]
    convolved = convolved / convolved.sum()
    
    # Scale to photon counts
    gt_curve = convolved / convolved.sum() * total_counts
    gt_curve_with_bg = gt_curve + background
    
    # Noisy measurement (Poisson)
    measured = rng.poisson(gt_curve_with_bg).astype(np.float64)
    
    params = {
        'tau1_true': tau1_true,
        'tau2_true': tau2_true,
        'a1_true': a1_true,
        'a2_true': a2_true,
        'background': background,
        'total_counts': total_counts,
        'n_channels': n_channels,
        'dt': dt,
        'irf_center': irf_center,
        'irf_fwhm': irf_fwhm,
    }
    
    return {
        'time': time,
        'irf': irf,
        'gt_curve': gt_curve,
        'gt_curve_with_bg': gt_curve_with_bg,
        'measured': measured,
        'params': params,
    }
