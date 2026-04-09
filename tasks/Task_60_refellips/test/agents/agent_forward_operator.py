import numpy as np

import matplotlib

matplotlib.use('Agg')

def cauchy_n(wavelength_nm, A, B, C):
    """Cauchy dispersion: n(λ) = A + B/λ² + C/λ⁴"""
    lam_um = wavelength_nm / 1000.0
    return A + B / lam_um**2 + C / lam_um**4

def si_optical_constants(wavelength_nm, n_si_633, k_si_633):
    """Approximate Si optical constants (simple dispersion)."""
    lam_um = wavelength_nm / 1000.0
    n = n_si_633 + 0.8 * (0.633 / lam_um - 1)
    k = k_si_633 * (0.633 / lam_um) ** 2
    return n, k

def forward_operator(params, wavelengths, angle_deg, n_si_633, k_si_633):
    """
    Compute ellipsometric Ψ and Δ for a thin film on Si substrate
    using the transfer matrix method.

    This implements the standard 2×2 transfer matrix for
    ambient/film/substrate with Fresnel coefficients.

    Parameters
    ----------
    params : dict
        Film parameters (thickness, A, B, C, k_amp).
    wavelengths : array
        Wavelengths [nm].
    angle_deg : float
        Angle of incidence [degrees].
    n_si_633 : float
        Si refractive index at 633 nm.
    k_si_633 : float
        Si extinction coefficient at 633 nm.

    Returns
    -------
    psi : array
        Ψ [degrees]
    delta : array
        Δ [degrees]
    """
    theta0 = np.radians(angle_deg)
    n0 = 1.0  # air

    psi = np.zeros(len(wavelengths))
    delta = np.zeros(len(wavelengths))

    for i, lam in enumerate(wavelengths):
        # Film optical constants
        n1 = cauchy_n(lam, params["A"], params["B"], params["C"])
        k1 = params["k_amp"] * (400.0 / lam) ** 2  # Urbach-like absorption
        N1 = n1 + 1j * k1

        # Substrate
        n2, k2 = si_optical_constants(lam, n_si_633, k_si_633)
        N2 = n2 + 1j * k2

        # Snell's law
        sin_theta0 = np.sin(theta0)
        cos_theta0 = np.cos(theta0)
        cos_theta1 = np.sqrt(1 - (n0 * sin_theta0 / N1) ** 2)
        cos_theta2 = np.sqrt(1 - (n0 * sin_theta0 / N2) ** 2)

        # Fresnel coefficients: air→film
        rp01 = (N1 * cos_theta0 - n0 * cos_theta1) / (N1 * cos_theta0 + n0 * cos_theta1)
        rs01 = (n0 * cos_theta0 - N1 * cos_theta1) / (n0 * cos_theta0 + N1 * cos_theta1)

        # Fresnel coefficients: film→substrate
        rp12 = (N2 * cos_theta1 - N1 * cos_theta2) / (N2 * cos_theta1 + N1 * cos_theta2)
        rs12 = (N1 * cos_theta1 - N2 * cos_theta2) / (N1 * cos_theta1 + N2 * cos_theta2)

        # Phase thickness
        beta = 2 * np.pi * params["thickness"] * N1 * cos_theta1 / lam

        # Total reflection coefficients (Airy formula)
        phase = np.exp(-2j * beta)
        Rp = (rp01 + rp12 * phase) / (1 + rp01 * rp12 * phase)
        Rs = (rs01 + rs12 * phase) / (1 + rs01 * rs12 * phase)

        # Ellipsometric ratio
        rho = Rp / Rs
        psi[i] = np.degrees(np.arctan(np.abs(rho)))
        delta[i] = np.degrees(np.angle(rho))

    return psi, delta
