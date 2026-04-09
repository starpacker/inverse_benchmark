import numpy as np

import matplotlib

matplotlib.use('Agg')

import batman

def load_and_preprocess_data(gt_params, n_time, t_span, noise_ppm, seed):
    """
    Generate synthetic transit light curve with batman.
    
    Parameters
    ----------
    gt_params : dict
        Ground-truth transit parameters.
    n_time : int
        Number of time points.
    t_span : float
        Observation window around transit [days].
    noise_ppm : float
        Photometric noise [ppm].
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    t : np.ndarray
        Time array [days].
    flux_noisy : np.ndarray
        Noisy flux measurements.
    flux_clean : np.ndarray
        Clean (ground truth) flux.
    flux_err : np.ndarray
        Error bars on flux.
    """
    print("[DATA] Generating synthetic transit light curve with batman ...")
    
    t = np.linspace(-t_span, t_span, n_time)
    
    # Generate clean light curve using forward operator
    flux_clean = forward_operator(gt_params, t)
    
    # Transit depth check
    depth = 1.0 - flux_clean.min()
    print(f"[DATA] Transit depth = {depth*1e6:.0f} ppm  "
          f"(Rp/Rs = {gt_params['rp']:.3f})")
    
    # Add Gaussian photometric noise
    rng = np.random.default_rng(seed)
    sigma = noise_ppm * 1e-6  # convert ppm to relative flux
    flux_noisy = flux_clean + sigma * rng.standard_normal(n_time)
    flux_err = np.full(n_time, sigma)
    
    print(f"[DATA] Noise = {noise_ppm} ppm  |  {n_time} points  |  "
          f"T ∈ [{t[0]:.3f}, {t[-1]:.3f}] days")
    
    return t, flux_noisy, flux_clean, flux_err

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
