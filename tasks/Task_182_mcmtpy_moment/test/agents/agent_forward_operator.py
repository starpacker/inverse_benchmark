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

def forward_operator(params, config, T=None, WIN_INDICES=None, windowed=False):
    """
    Compute synthetic P-wave waveforms at all stations.
    
    Parameters
    ----------
    params : tuple or array
        (strike, dip, rake, log_M0) source parameters
    config : dict
        Configuration parameters
    T : array, optional
        Time array (required if windowed=False)
    WIN_INDICES : list, optional
        Window indices (required if windowed=True)
    windowed : bool
        If True, compute only within signal windows
        
    Returns
    -------
    waveforms : ndarray or list
        Synthetic waveforms. If windowed=False, returns (N_STATIONS x NT) array.
        If windowed=True, returns list of windowed arrays.
    """
    strike, dip, rake, log_M0 = params
    M0 = 10.0 ** log_M0
    
    VP = config['VP']
    N_STATIONS = config['N_STATIONS']
    AZIMUTHS = config['AZIMUTHS']
    DISTANCES = config['DISTANCES']
    TAKEOFFS = config['TAKEOFFS']
    STF_WIDTH = config['STF_WIDTH']
    
    if windowed:
        result = []
        for i in range(N_STATIONS):
            R = radiation_P(strike, dip, rake, AZIMUTHS[i], TAKEOFFS[i])
            travel_time = DISTANCES[i] / VP
            amp = R * M0 / DISTANCES[i]
            i0, i1 = WIN_INDICES[i]
            t_win = T[i0:i1]
            stf = source_time_function(t_win, travel_time, half_width=STF_WIDTH)
            result.append(amp * stf)
        return result
    else:
        NT = len(T)
        waveforms = np.zeros((N_STATIONS, NT))
        for i in range(N_STATIONS):
            R = radiation_P(strike, dip, rake, AZIMUTHS[i], TAKEOFFS[i])
            travel_time = DISTANCES[i] / VP
            amp = R * M0 / DISTANCES[i]
            stf = source_time_function(T, travel_time, half_width=STF_WIDTH)
            waveforms[i] = amp * stf
        return waveforms
