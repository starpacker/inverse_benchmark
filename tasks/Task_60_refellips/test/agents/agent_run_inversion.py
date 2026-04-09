import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.optimize import differential_evolution, minimize

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

def run_inversion(data, seed):
    """
    Fit ellipsometric parameters using DE + Nelder-Mead.

    Free parameters: thickness, A, B, C, k_amp

    Parameters
    ----------
    data : dict
        Dictionary containing wavelengths, noisy measurements, and other parameters.
    seed : int
        Random seed for differential evolution.

    Returns
    -------
    result : dict
        Dictionary containing fit_params, psi_fit, delta_fit.
    """
    wavelengths = data["wavelengths"]
    psi_meas = data["psi_noisy"]
    delta_meas = data["delta_noisy"]
    angle_of_incidence = data["angle_of_incidence"]
    noise_level = data["noise_level"]
    n_si_633 = data["n_si_633"]
    k_si_633 = data["k_si_633"]

    def cost(x):
        thick, A, B, C, k_amp = x
        params = {"thickness": thick, "A": A, "B": B, "C": C, "k_amp": k_amp}
        try:
            psi_calc, delta_calc = forward_operator(
                params, wavelengths, angle_of_incidence, n_si_633, k_si_633
            )
            res_psi = (psi_meas - psi_calc) / noise_level
            res_delta = (delta_meas - delta_calc) / noise_level
            return np.sum(res_psi**2 + res_delta**2)
        except Exception:
            return 1e20

    bounds = [
        (10, 500),       # thickness [nm]
        (1.3, 2.0),      # A
        (0.0, 0.05),     # B
        (0.0, 0.005),    # C
        (0.0, 0.1),      # k_amp
    ]

    print("[RECON] Stage 1 — Differential Evolution ...")
    result_de = differential_evolution(
        cost, bounds, seed=seed, maxiter=150, tol=1e-5, popsize=15
    )
    print(f"[RECON]   χ² = {result_de.fun:.2f}")

    print("[RECON] Stage 2 — Nelder-Mead refinement ...")
    result_nm = minimize(
        cost, result_de.x, method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-6}
    )
    print(f"[RECON]   χ² = {result_nm.fun:.2f}")

    thick, A, B, C, k_amp = result_nm.x
    fit_params = {
        "thickness": float(thick),
        "A": float(A),
        "B": float(B),
        "C": float(C),
        "k_amp": float(k_amp)
    }
    psi_fit, delta_fit = forward_operator(
        fit_params, wavelengths, angle_of_incidence, n_si_633, k_si_633
    )

    result = {
        "fit_params": fit_params,
        "psi_fit": psi_fit,
        "delta_fit": delta_fit,
    }

    return result
