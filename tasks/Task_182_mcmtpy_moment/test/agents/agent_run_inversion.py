import numpy as np

import emcee

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

def run_inversion(data, mcmc_config):
    """
    Run MCMC inversion to recover source parameters.
    
    Parameters
    ----------
    data : dict
        Data dictionary from load_and_preprocess_data
    mcmc_config : dict
        MCMC configuration with NWALKERS, NSTEPS, BURNIN, p0_centre
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - map_estimate: MAP parameter estimates
        - median_estimate: median parameter estimates
        - flat_samples: flattened MCMC samples after burn-in
        - chain: full MCMC chain
        - log_probs: log probabilities
        - sampler: emcee sampler object
    """
    d_obs_win = data['d_obs_win']
    WIN_INDICES = data['WIN_INDICES']
    T = data['T']
    sigma_noise = data['sigma_noise']
    config = data['config']
    N_STATIONS = config['N_STATIONS']
    
    NWALKERS = mcmc_config['NWALKERS']
    NSTEPS = mcmc_config['NSTEPS']
    BURNIN = mcmc_config['BURNIN']
    p0_centre = mcmc_config['p0_centre']
    NDIM = 4
    
    def log_prior(theta):
        strike, dip, rake, log_M0 = theta
        if 0 <= strike <= 360 and 0 <= dip <= 90 and -180 <= rake <= 180 and 14 <= log_M0 <= 18:
            return 0.0
        return -np.inf
    
    def log_likelihood(theta):
        strike, dip, rake, log_M0 = theta
        d_syn_win = forward_operator(
            (strike, dip, rake, log_M0), 
            config, 
            T=T, 
            WIN_INDICES=WIN_INDICES, 
            windowed=True
        )
        chi2 = 0.0
        for i in range(N_STATIONS):
            residual = d_obs_win[i] - d_syn_win[i]
            chi2 += np.sum(residual**2) / sigma_noise**2
        return -0.5 * chi2
    
    def log_posterior(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll
    
    # Initialize walkers
    p0 = p0_centre + 0.1 * np.random.randn(NWALKERS, NDIM)
    p0[:, 0] = np.clip(p0[:, 0], 0.5, 359.5)
    p0[:, 1] = np.clip(p0[:, 1], 0.5, 89.5)
    p0[:, 2] = np.clip(p0[:, 2], -179.5, 179.5)
    p0[:, 3] = np.clip(p0[:, 3], 14.1, 17.9)
    
    print("[INFO] Running MCMC …")
    sampler = emcee.EnsembleSampler(NWALKERS, NDIM, log_posterior)
    sampler.run_mcmc(p0, NSTEPS, progress=False)
    
    chain = sampler.get_chain()
    flat = sampler.get_chain(discard=BURNIN, flat=True)
    log_probs = sampler.get_log_prob(discard=BURNIN, flat=True)
    
    idx_map = np.argmax(log_probs)
    map_est = flat[idx_map]
    median_est = np.median(flat, axis=0)
    
    print(f"[RESULT] MAP:    strike={map_est[0]:.2f}, dip={map_est[1]:.2f}, "
          f"rake={map_est[2]:.2f}, log10(M0)={map_est[3]:.4f}")
    print(f"[RESULT] Median: strike={median_est[0]:.2f}, dip={median_est[1]:.2f}, "
          f"rake={median_est[2]:.2f}, log10(M0)={median_est[3]:.4f}")
    
    result = {
        'map_estimate': map_est,
        'median_estimate': median_est,
        'flat_samples': flat,
        'chain': chain,
        'log_probs': log_probs,
        'sampler': sampler
    }
    
    return result
