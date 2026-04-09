import numpy as np

import matplotlib

matplotlib.use("Agg")

def source_time_function(t, t0, half_width=0.2):
    """Gaussian source-time function centred at t0."""
    return np.exp(-((t - t0) ** 2) / (2.0 * half_width ** 2))

def radiation_P(strike_deg, dip_deg, rake_deg, azimuth_deg, takeoff_deg):
    """
    P-wave radiation pattern for a double-couple source.
    Aki & Richards (2002), Eq. 4.29.
    """
    s = np.radians(strike_deg)
    d = np.radians(dip_deg)
    r = np.radians(rake_deg)
    az = np.radians(azimuth_deg)
    ih = np.radians(takeoff_deg)
    phi = az - s

    R = (np.cos(r) * np.sin(d) * np.sin(ih)**2 * np.sin(2 * phi)
         - np.cos(r) * np.cos(d) * np.sin(2 * ih) * np.cos(phi)
         + np.sin(r) * np.sin(2 * d) * (np.cos(ih)**2 - np.sin(ih)**2 * np.sin(phi)**2)
         + np.sin(r) * np.cos(2 * d) * np.sin(2 * ih) * np.sin(phi))
    return R

def load_and_preprocess_data(gt_strike, gt_dip, gt_rake, gt_log_m0, config):
    """
    Generate synthetic seismic waveform data with noise and preprocess.
    
    Parameters
    ----------
    gt_strike : float
        Ground truth strike angle (degrees)
    gt_dip : float
        Ground truth dip angle (degrees)
    gt_rake : float
        Ground truth rake angle (degrees)
    gt_log_m0 : float
        Ground truth log10 of seismic moment
    config : dict
        Configuration parameters including VP, DT, T_MAX, STF_WIDTH, 
        N_STATIONS, AZIMUTHS, DISTANCES, TAKEOFFS, NOISE_FRAC
        
    Returns
    -------
    data : dict
        Dictionary containing:
        - d_obs: observed waveforms with noise (N_STATIONS x NT)
        - d_clean: clean waveforms without noise (N_STATIONS x NT)
        - d_obs_win: windowed observed data (list of arrays)
        - WIN_INDICES: signal window indices for each station
        - T: time array
        - sigma_noise: noise standard deviation
        - config: configuration parameters
        - ground_truth: dict of ground truth parameters
    """
    VP = config['VP']
    DT = config['DT']
    T_MAX = config['T_MAX']
    NT = int(T_MAX / DT) + 1
    T = np.linspace(0, T_MAX, NT)
    STF_WIDTH = config['STF_WIDTH']
    N_STATIONS = config['N_STATIONS']
    AZIMUTHS = config['AZIMUTHS']
    DISTANCES = config['DISTANCES']
    TAKEOFFS = config['TAKEOFFS']
    NOISE_FRAC = config['NOISE_FRAC']
    
    # Generate clean synthetic data using forward model
    M0 = 10.0 ** gt_log_m0
    d_clean = np.zeros((N_STATIONS, NT))
    for i in range(N_STATIONS):
        R = radiation_P(gt_strike, gt_dip, gt_rake, AZIMUTHS[i], TAKEOFFS[i])
        travel_time = DISTANCES[i] / VP
        amp = R * M0 / DISTANCES[i]
        stf = source_time_function(T, travel_time, half_width=STF_WIDTH)
        d_clean[i] = amp * stf
    
    # Add noise
    max_amp = np.max(np.abs(d_clean))
    sigma_noise = NOISE_FRAC * max_amp
    d_obs = d_clean + sigma_noise * np.random.randn(*d_clean.shape)
    
    # Compute signal windows
    WIN_HALF = int(5 * STF_WIDTH / DT)
    WIN_INDICES = []
    for i in range(N_STATIONS):
        tt_idx = int(DISTANCES[i] / VP / DT)
        i0 = max(0, tt_idx - WIN_HALF)
        i1 = min(NT, tt_idx + WIN_HALF)
        WIN_INDICES.append((i0, i1))
    
    # Extract windowed observed data
    d_obs_win = [d_obs[i, i0:i1].copy() for i, (i0, i1) in enumerate(WIN_INDICES)]
    
    print(f"[INFO] GT: strike={gt_strike}, dip={gt_dip}, rake={gt_rake}, M0=1e{gt_log_m0:.0f}")
    print(f"[INFO] sigma_noise = {sigma_noise:.4e}")
    
    data = {
        'd_obs': d_obs,
        'd_clean': d_clean,
        'd_obs_win': d_obs_win,
        'WIN_INDICES': WIN_INDICES,
        'T': T,
        'NT': NT,
        'sigma_noise': sigma_noise,
        'config': config,
        'ground_truth': {
            'strike': gt_strike,
            'dip': gt_dip,
            'rake': gt_rake,
            'log_m0': gt_log_m0
        }
    }
    
    return data
