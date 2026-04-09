import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.optimize import differential_evolution, minimize

import batman

def forward_operator(params, t):
    """
    Compute transit light curve F(t) using batman.

    batman implements the Mandel & Agol (2002) analytic model
    for planetary transit light curves, supporting:
      - Uniform, linear, quadratic, nonlinear limb darkening
      - Eccentric orbits
      - Secondary eclipses

    Parameters
    ----------
    params : dict
        Transit parameters containing:
        - rp: planet radius / stellar radius (Rp/Rs)
        - a: semi-major axis / stellar radius (a/Rs)
        - inc: orbital inclination [degrees]
        - ecc: eccentricity
        - w: argument of periastron [degrees]
        - t0: mid-transit time [days]
        - per: orbital period [days]
        - u1, u2: quadratic limb-darkening coefficients
    t : np.ndarray
        Time array [days].

    Returns
    -------
    flux : np.ndarray
        Normalised flux F(t).
    """
    bm_params = batman.TransitParams()
    bm_params.rp = params["rp"]
    bm_params.a = params["a"]
    bm_params.inc = params["inc"]
    bm_params.ecc = params["ecc"]
    bm_params.w = params["w"]
    bm_params.t0 = params["t0"]
    bm_params.per = params["per"]
    bm_params.u = [params["u1"], params["u2"]]
    bm_params.limb_dark = "quadratic"

    model = batman.TransitModel(bm_params, t)
    flux = model.light_curve(bm_params)
    return flux

def run_inversion(t, flux_meas, flux_err, fixed_params, seed):
    """
    Fit transit parameters using DE + Nelder-Mead through batman forward.

    Free parameters: rp, a, inc, u1, u2
    Fixed parameters: t0, per, ecc, w (assumed known from ephemeris)

    Parameters
    ----------
    t : np.ndarray
        Time array.
    flux_meas : np.ndarray
        Measured (noisy) flux.
    flux_err : np.ndarray
        Error bars.
    fixed_params : dict
        Fixed parameters (t0, per, ecc, w).
    seed : int
        Random seed for optimization.

    Returns
    -------
    fit_params : dict
        Best-fit parameter values.
    flux_fit : np.ndarray
        Best-fit light curve.
    """
    n_time = len(t)
    
    def chi2(x):
        rp, a, inc, u1, u2 = x
        params = {
            "rp": rp, "a": a, "inc": inc,
            "u1": u1, "u2": u2, **fixed_params
        }
        try:
            model_flux = forward_operator(params, t)
        except Exception:
            return 1e20
        return np.sum(((flux_meas - model_flux) / flux_err) ** 2)

    # Bounds for free parameters
    bounds = [
        (0.01, 0.3),    # rp  (Rp/Rs)
        (2.0, 50.0),    # a   (a/Rs)
        (70.0, 90.0),   # inc [deg]
        (0.0, 0.8),     # u1
        (-0.3, 0.6),    # u2
    ]

    # Stage 1: Differential Evolution
    print("[RECON] Stage 1 — Differential Evolution (global search) ...")
    result_de = differential_evolution(
        chi2, bounds, seed=seed,
        maxiter=150, tol=1e-5, popsize=15,
        mutation=(0.5, 1.5), recombination=0.8
    )
    print(f"[RECON]   χ² = {result_de.fun:.2f}  "
          f"(reduced χ²/ν = {result_de.fun/n_time:.4f})")

    # Stage 2: Nelder-Mead local refinement
    print("[RECON] Stage 2 — Nelder-Mead refinement ...")
    result_nm = minimize(
        chi2, result_de.x, method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6}
    )
    print(f"[RECON]   χ² = {result_nm.fun:.2f}")

    rp, a, inc, u1, u2 = result_nm.x
    fit_params = {
        "rp": float(rp), "a": float(a), "inc": float(inc),
        "u1": float(u1), "u2": float(u2), **fixed_params
    }

    flux_fit = forward_operator(fit_params, t)
    return fit_params, flux_fit
