import numpy as np

import matplotlib

matplotlib.use("Agg")

def load_and_preprocess_data(n_time, dt, tau_peak, tau_sigma, snr, seed):
    """
    Generate synthetic AGN light curves and ground-truth transfer function.
    
    Parameters:
    -----------
    n_time : int
        Number of time samples
    dt : float
        Time step in days
    tau_peak : float
        Peak lag of transfer function in days
    tau_sigma : float
        Width of transfer function in days
    snr : float
        Signal-to-noise ratio
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict containing:
        - t: time array
        - continuum: AGN continuum light curve
        - tau: lag array
        - psi_gt: ground-truth transfer function
        - line_clean: clean emission line light curve
        - line_obs: observed (noisy) emission line light curve
        - noise_std: standard deviation of added noise
        - dt: time step
        - tau_peak: peak lag
        - tau_sigma: transfer function width
    """
    np.random.seed(seed)
    
    # Generate continuum using damped random walk (DRW) process
    tau_drw = 100.0   # DRW timescale (days)
    sigma_drw = 0.15  # DRW amplitude
    t = np.arange(n_time) * dt
    x = np.zeros(n_time)
    for i in range(1, n_time):
        decay = np.exp(-dt / tau_drw)
        x[i] = decay * x[i-1] + sigma_drw * np.sqrt(1 - decay**2) * np.random.randn()
    continuum = 1.0 + x
    
    # Generate ground-truth transfer function: Gaussian in lag space
    tau = np.arange(n_time) * dt
    psi_gt = np.exp(-0.5 * ((tau - tau_peak) / tau_sigma)**2)
    psi_gt = psi_gt / (psi_gt.sum() * dt)  # normalise to unit integral
    
    # Generate emission line via convolution
    line_clean = np.convolve(continuum, psi_gt * dt, mode='full')[:n_time]
    noise_std = line_clean.std() / snr
    noise = np.random.normal(0, noise_std, n_time)
    line_obs = line_clean + noise
    
    return {
        't': t,
        'continuum': continuum,
        'tau': tau,
        'psi_gt': psi_gt,
        'line_clean': line_clean,
        'line_obs': line_obs,
        'noise_std': noise_std,
        'dt': dt,
        'tau_peak': tau_peak,
        'tau_sigma': tau_sigma,
        'n_time': n_time
    }
