import numpy as np

import matplotlib

matplotlib.use("Agg")

C_LIGHT = 2.998e18

H_PLANCK = 6.626e-27

K_BOLTZ = 1.381e-16

L_SUN = 3.828e33

PC_CM = 3.086e18

DIST_CM = 10.0 * PC_CM

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
