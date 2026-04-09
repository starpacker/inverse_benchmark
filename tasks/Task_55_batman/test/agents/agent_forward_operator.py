import matplotlib

matplotlib.use('Agg')

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
