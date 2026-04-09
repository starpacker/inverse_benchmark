import time

import numpy as np

import matplotlib

matplotlib.use("Agg")

import emcee

C_LIGHT = 2.998e18

H_PLANCK = 6.626e-27

K_BOLTZ = 1.381e-16

L_SUN = 3.828e33

PC_CM = 3.086e18

DIST_CM = 10.0 * PC_CM

FILTER_WAVES = np.array([3551.0, 4686.0, 6166.0, 7480.0, 8932.0,
                          12350.0, 16620.0, 21590.0])

FILTER_WIDTHS = np.array([560.0, 1380.0, 1370.0, 1530.0, 950.0,
                           1620.0, 2510.0, 2620.0])

RV_CALZETTI = 4.05

def calzetti_kprime(wave_um):
    """Calzetti et al. (2000) attenuation curve k'(lambda).
    wave_um : wavelength in microns.
    Returns k'(lambda) such that A(lambda) = Av / Rv * k'(lambda).
    """
    k = np.zeros_like(wave_um)
    lo = (wave_um >= 0.12) & (wave_um < 0.63)
    hi = (wave_um >= 0.63) & (wave_um <= 2.20)

    k[lo] = (2.659 * (-2.156 + 1.509 / wave_um[lo]
              - 0.198 / wave_um[lo]**2
              + 0.011 / wave_um[lo]**3) + RV_CALZETTI)
    k[hi] = (2.659 * (-1.857 + 1.040 / wave_um[hi]) + RV_CALZETTI)
    k = np.clip(k, 0.0, None)
    return k

def dust_attenuation(wave_aa, Av):
    """Return flux attenuation factor (multiply flux by this) for given Av."""
    wave_um = wave_aa / 1e4
    kp = calzetti_kprime(wave_um)
    tau = Av * kp / RV_CALZETTI
    return 10.0 ** (-0.4 * tau)

def effective_temperature(log_age, metallicity):
    """Simple mapping from age+Z to an effective temperature for the
    composite stellar population.  This is a toy analytic approximation:
      T_eff ~ 5000 * (age/1e9)^{-0.15} * (Z/0.02)^{0.05}  K
    Young, metal-poor populations are hotter.
    """
    age_gyr = 10.0 ** (log_age - 9.0)  # age in Gyr
    T = 5500.0 * age_gyr ** (-0.18) * (metallicity / 0.02) ** 0.05
    return np.clip(T, 2500.0, 50000.0)

def composite_spectrum(wave_aa, log_mass, log_age, metallicity, Av):
    """Compute observed flux density F_nu (erg/s/cm^2/Hz) at 10 pc
    for the analytic SED model.

    The model is a two-component modified blackbody:
      - Hot component (young stars): T from age/Z mapping
      - Cool component: 0.6 * T  (older underlying population)
    Relative luminosity fraction: hot=0.7, cool=0.3 (varies with age).
    """
    mass = 10.0 ** log_mass  # solar masses
    T_hot = effective_temperature(log_age, metallicity)
    T_cool = 0.55 * T_hot

    # Younger populations have larger hot fraction
    age_gyr = 10.0 ** (log_age - 9.0)
    f_hot = np.clip(0.8 - 0.15 * np.log10(age_gyr + 0.01), 0.2, 0.95)

    # Planck function B_nu(T) in erg/s/cm^2/Hz/sr
    nu = C_LIGHT / wave_aa
    def planck_nu(T):
        x = H_PLANCK * nu / (K_BOLTZ * T)
        x = np.clip(x, 0, 500)  # prevent overflow
        return 2 * H_PLANCK * nu**3 / C_LIGHT**2 / (np.exp(x) - 1.0 + 1e-30)

    B_hot = planck_nu(T_hot)
    B_cool = planck_nu(T_cool)

    # Composite (arbitrary normalisation, will be scaled by mass)
    B_composite = f_hot * B_hot + (1.0 - f_hot) * B_cool

    # Mass-to-light scaling: L ~ mass * L_sun (simplified)
    sigma_SB = 5.670e-5  # erg/cm^2/s/K^4
    L_bol_per_unit = sigma_SB * (f_hot * T_hot**4 + (1 - f_hot) * T_cool**4)
    L_target = mass * L_SUN / (4.0 * np.pi * DIST_CM**2)

    # Scale spectrum
    scale = L_target / (np.pi * L_bol_per_unit + 1e-30)
    F_nu = scale * B_composite

    # Apply dust attenuation
    atten = dust_attenuation(wave_aa, Av)
    F_nu *= atten

    return F_nu

def forward_operator(params, filter_waves=None, filter_widths=None):
    """
    Compute model photometry in 8 bands given parameter vector.
    
    This is the forward model that maps physical parameters to observable fluxes.
    
    Parameters
    ----------
    params : array-like
        Parameter vector [log_mass, log_age, metallicity, Av]
    filter_waves : ndarray, optional
        Filter central wavelengths in Angstroms (default: FILTER_WAVES)
    filter_widths : ndarray, optional
        Filter bandwidths in Angstroms (default: FILTER_WIDTHS)
    
    Returns
    -------
    ndarray
        Array of flux densities (erg/s/cm^2/Hz) for each band
    """
    if filter_waves is None:
        filter_waves = FILTER_WAVES
    if filter_widths is None:
        filter_widths = FILTER_WIDTHS
    
    log_mass, log_age, metallicity, Av = params
    fluxes = np.zeros(len(filter_waves))
    
    for i in range(len(filter_waves)):
        # Integrate over filter bandwidth (trapezoidal, 50 points)
        w_lo = filter_waves[i] - filter_widths[i] / 2.0
        w_hi = filter_waves[i] + filter_widths[i] / 2.0
        wave_grid = np.linspace(w_lo, w_hi, 50)
        spec = composite_spectrum(wave_grid, log_mass, log_age, metallicity, Av)
        fluxes[i] = np.trapz(spec, wave_grid) / (w_hi - w_lo)
    
    return fluxes

def run_inversion(data, nwalkers=24, nsteps=800, nburn=300):
    """
    Run MCMC inversion to recover SED parameters from observed photometry.
    
    Parameters
    ----------
    data : dict
        Data dictionary from load_and_preprocess_data containing:
        - 'obs_flux': observed flux values
        - 'obs_unc': flux uncertainties
        - 'true_params': true parameter values (for initialization)
        - 'param_bounds': parameter bounds
    nwalkers : int
        Number of MCMC walkers
    nsteps : int
        Number of MCMC steps
    nburn : int
        Number of burn-in steps to discard
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'chain': posterior samples after burn-in
        - 'full_chain': full chain including burn-in (nsteps, nwalkers, ndim)
        - 'median_params': median parameter values
        - 'std_params': standard deviation of parameters
        - 'best_params': maximum posterior parameters
        - 'log_probs': log probability values
        - 'acceptance_fraction': MCMC acceptance fraction
        - 'elapsed_time': time taken for MCMC
        - 'sampler': the emcee sampler object
    """
    obs_flux = data['obs_flux']
    obs_unc = data['obs_unc']
    true_params = data['true_params']
    param_bounds = data['param_bounds']
    
    ndim = len(true_params)
    
    # Define log prior
    def log_prior(params):
        for i, (lo, hi) in enumerate(param_bounds):
            if params[i] < lo or params[i] > hi:
                return -np.inf
        return 0.0
    
    # Define log likelihood
    def log_likelihood(params):
        model = forward_operator(params)
        if np.any(model <= 0):
            return -np.inf
        chi2 = np.sum(((obs_flux - model) / obs_unc) ** 2)
        return -0.5 * chi2
    
    # Define log probability
    def log_probability(params):
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        ll = log_likelihood(params)
        return lp + ll
    
    # Initialize walkers near truth with small scatter
    np.random.seed(42)
    p0 = np.array([true_params + 0.05 * np.random.randn(ndim) *
                    (param_bounds[:, 1] - param_bounds[:, 0])
                    for _ in range(nwalkers)])
    
    # Clip to bounds
    for i in range(ndim):
        p0[:, i] = np.clip(p0[:, i], param_bounds[i, 0] + 1e-6,
                            param_bounds[i, 1] - 1e-6)
    
    # Create sampler and run MCMC
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    
    t0 = time.time()
    sampler.run_mcmc(p0, nsteps, progress=True)
    elapsed = time.time() - t0
    
    # Extract results
    chain = sampler.get_chain(discard=nburn, flat=True)
    full_chain = sampler.get_chain()
    log_probs = sampler.get_log_prob(discard=nburn, flat=True)
    
    median_params = np.median(chain, axis=0)
    std_params = np.std(chain, axis=0)
    best_idx = np.argmax(log_probs)
    best_params = chain[best_idx]
    
    acc_frac = np.mean(sampler.acceptance_fraction)
    
    result = {
        'chain': chain,
        'full_chain': full_chain,
        'median_params': median_params,
        'std_params': std_params,
        'best_params': best_params,
        'log_probs': log_probs,
        'acceptance_fraction': acc_frac,
        'elapsed_time': elapsed,
        'sampler': sampler,
        'nwalkers': nwalkers,
        'nsteps': nsteps,
        'nburn': nburn,
    }
    
    return result
