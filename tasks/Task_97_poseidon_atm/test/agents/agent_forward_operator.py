import numpy as np

import matplotlib

matplotlib.use('Agg')

K_B = 1.380649e-23

AMU = 1.66054e-27

G = 6.674e-11

MU_ATM = 2.3 * AMU

SIGMA_RAY_REF = 5.31e-31

WAV_RAY_REF = 0.35e-6

SPECIES_BANDS = {
    "H2O": [
        (1.4e-6,  0.15e-6,  1.0e-25),
        (1.85e-6, 0.12e-6,  6.0e-26),
        (2.7e-6,  0.20e-6,  1.5e-25),
    ],
    "CH4": [
        (1.65e-6, 0.10e-6,  8.0e-26),
        (2.3e-6,  0.15e-6,  5.0e-26),
        (3.3e-6,  0.25e-6,  1.2e-25),
    ],
    "CO2": [
        (4.3e-6,  0.20e-6,  2.0e-25),
        (2.0e-6,  0.08e-6,  2.0e-26),
    ],
}

def compute_cross_section(wavelengths, species_name):
    """
    Compute simplified absorption cross-section for a species.
    Uses sum of Gaussian absorption bands.
    """
    bands = SPECIES_BANDS[species_name]
    sigma = np.zeros_like(wavelengths)
    for center, width, peak in bands:
        sigma += peak * np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
    return sigma

def compute_rayleigh(wavelengths):
    """
    Rayleigh scattering cross-section for H₂ (λ⁻⁴ dependence).
    """
    return SIGMA_RAY_REF * (WAV_RAY_REF / wavelengths) ** 4

def forward_operator(params, wavelengths, r_star, m_planet, n_layers, p_top, p_bottom):
    """
    Compute the transmission spectrum (transit depth vs wavelength).

    Implements a simplified version of the atmospheric transmission
    calculation as in POSEIDON/Exo-Transmit/petitRADTRANS:

    1. Build pressure-altitude grid assuming hydrostatic equilibrium.
    2. At each layer, compute number densities of absorbers.
    3. For each wavelength, compute slant optical depth through each
       annular ring of atmosphere.
    4. Integrate to get the effective transit radius R_eff(λ).
    5. Transit depth D(λ) = (R_eff(λ)/R_star)².

    Parameters
    ----------
    params : dict
        Atmospheric parameters containing T, log_X_H2O, log_X_CH4, log_X_CO2, R_p.
    wavelengths : np.ndarray
        Wavelength array [m].
    r_star : float
        Stellar radius [m].
    m_planet : float
        Planet mass [kg].
    n_layers : int
        Number of atmospheric layers.
    p_top : float
        Top pressure [Pa].
    p_bottom : float
        Bottom pressure [Pa].

    Returns
    -------
    transit_depth : np.ndarray
        Transit depth D(λ).
    """
    T = params["T"]
    X_H2O = 10.0 ** params["log_X_H2O"]
    X_CH4 = 10.0 ** params["log_X_CH4"]
    X_CO2 = 10.0 ** params["log_X_CO2"]
    R_p = params["R_p"]

    # Pressure grid (log-spaced, top to bottom)
    pressures = np.logspace(np.log10(p_top), np.log10(p_bottom), n_layers)

    # Scale height H = k_B T / (mu g)
    g = G * m_planet / R_p ** 2  # surface gravity
    H = K_B * T / (MU_ATM * g)

    # Altitude grid from hydrostatic equilibrium: z = -H ln(P/P_ref)
    P_ref = pressures[-1]  # reference at bottom
    altitudes = -H * np.log(pressures / P_ref)  # z=0 at bottom

    # Layer boundaries (midpoints between levels)
    alt_boundaries = np.zeros(n_layers + 1)
    alt_boundaries[0] = altitudes[0] + 0.5 * (altitudes[0] - altitudes[1])
    alt_boundaries[-1] = altitudes[-1] - 0.5 * (altitudes[-2] - altitudes[-1])
    alt_boundaries[1:-1] = 0.5 * (altitudes[:-1] + altitudes[1:])
    dz = np.abs(np.diff(alt_boundaries))  # layer thicknesses

    # Number densities [m⁻³]
    n_total = pressures / (K_B * T)
    n_H2O = X_H2O * n_total
    n_CH4 = X_CH4 * n_total
    n_CO2 = X_CO2 * n_total

    # Cross-sections
    sigma_H2O = compute_cross_section(wavelengths, "H2O")
    sigma_CH4 = compute_cross_section(wavelengths, "CH4")
    sigma_CO2 = compute_cross_section(wavelengths, "CO2")
    sigma_ray = compute_rayleigh(wavelengths)

    # Effective radius calculation
    r = R_p + altitudes  # radius of each layer center

    # Transit depth: D(λ) = [R_p² + 2∫ r(1-exp(-τ)) dr] / R_star²
    transit_depth = np.zeros(len(wavelengths))

    for j in range(n_layers):
        # Total extinction at layer j for each wavelength
        kappa = (n_H2O[j] * sigma_H2O +
                 n_CH4[j] * sigma_CH4 +
                 n_CO2[j] * sigma_CO2 +
                 n_total[j] * sigma_ray)  # shape (N_wave,)

        # Slant path length through annulus at impact parameter b = r[j]
        ds = 2.0 * np.sqrt(2.0 * r[j] * H)

        # Optical depth along slant
        tau = kappa * ds  # shape (N_wave,)

        # Contribution to effective area
        transit_depth += 2.0 * r[j] * dz[j] * (1.0 - np.exp(-tau))

    # Add opaque disk of planet
    transit_depth = (R_p ** 2 + transit_depth) / r_star ** 2

    return transit_depth
