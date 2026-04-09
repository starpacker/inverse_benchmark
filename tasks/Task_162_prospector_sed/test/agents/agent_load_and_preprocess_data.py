import numpy as np

import matplotlib

matplotlib.use("Agg")

C_LIGHT = 2.998e18

H_PLANCK = 6.626e-27

K_BOLTZ = 1.381e-16

L_SUN = 3.828e33

PC_CM = 3.086e18

DIST_CM = 10.0 * PC_CM

FILTER_NAMES = ["u", "g", "r", "i", "z", "J", "H", "K"]

FILTER_WAVES = np.array([3551.0, 4686.0, 6166.0, 7480.0, 8932.0,
                          12350.0, 16620.0, 21590.0])

FILTER_WIDTHS = np.array([560.0, 1380.0, 1370.0, 1530.0, 950.0,
                           1620.0, 2510.0, 2620.0])

RV_CALZETTI = 4.05

PARAM_NAMES = ["log_mass", "log_age", "metallicity", "Av"]

PARAM_BOUNDS = np.array([
    [8.0, 12.0],     # log_mass
    [8.0, 10.2],     # log_age (100 Myr – 16 Gyr)
    [0.001, 0.05],   # metallicity
    [0.0, 3.0],      # Av
])

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

def load_and_preprocess_data(true_params, snr=20.0, seed=42):
    """
    Generate synthetic photometry data with known parameters.
    
    Parameters
    ----------
    true_params : ndarray
        Ground truth parameters [log_mass, log_age, metallicity, Av]
    snr : float
        Signal-to-noise ratio for synthetic observations
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'obs_flux': observed flux with noise
        - 'obs_unc': flux uncertainties
        - 'gt_flux': ground truth flux (noise-free)
        - 'true_params': true parameter values
        - 'filter_waves': filter wavelengths in Angstroms
        - 'filter_widths': filter bandwidths in Angstroms
        - 'filter_names': filter names
        - 'param_bounds': parameter bounds for MCMC
        - 'param_names': parameter names
    """
    np.random.seed(seed)
    
    # Compute ground truth flux from true parameters
    gt_flux = np.zeros(len(FILTER_WAVES))
    for i in range(len(FILTER_WAVES)):
        w_lo = FILTER_WAVES[i] - FILTER_WIDTHS[i] / 2.0
        w_hi = FILTER_WAVES[i] + FILTER_WIDTHS[i] / 2.0
        wave_grid = np.linspace(w_lo, w_hi, 50)
        spec = composite_spectrum(wave_grid, true_params[0], true_params[1], 
                                   true_params[2], true_params[3])
        gt_flux[i] = np.trapz(spec, wave_grid) / (w_hi - w_lo)
    
    # Add noise
    obs_unc = gt_flux / snr
    obs_flux = gt_flux + obs_unc * np.random.randn(len(gt_flux))
    
    data = {
        'obs_flux': obs_flux,
        'obs_unc': obs_unc,
        'gt_flux': gt_flux,
        'true_params': true_params,
        'filter_waves': FILTER_WAVES.copy(),
        'filter_widths': FILTER_WIDTHS.copy(),
        'filter_names': FILTER_NAMES.copy(),
        'param_bounds': PARAM_BOUNDS.copy(),
        'param_names': PARAM_NAMES.copy(),
    }
    
    return data
