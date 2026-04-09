import numpy as np

import matplotlib

matplotlib.use('Agg')

from refnx.reflect import SLD as SLDobj, ReflectModel, Structure

def build_structure(params, vary=False, dq_over_q=0.05):
    """
    Build a refnx Structure from a parameter dict.

    Parameters
    ----------
    params : dict
        Keys: polymer_thick, polymer_sld, polymer_rough,
              sio2_thick, sio2_sld, sio2_rough,
              si_sld, si_rough, bkg
    vary : bool
        If True, set bounds and mark parameters as variable.
    dq_over_q : float
        Resolution smearing parameter.

    Returns
    -------
    structure : refnx.reflect.Structure
    model : refnx.reflect.ReflectModel
    slabs : dict of slab objects
    """
    air = SLDobj(0.0, name='air')
    polymer = SLDobj(params["polymer_sld"], name='polymer')
    sio2 = SLDobj(params["sio2_sld"], name='sio2')
    si = SLDobj(params["si_sld"], name='si')

    polymer_slab = polymer(params["polymer_thick"], params["polymer_rough"])
    sio2_slab = sio2(params["sio2_thick"], params["sio2_rough"])
    si_slab = si(0, params["si_rough"])

    if vary:
        polymer_slab.thick.setp(bounds=(50, 500), vary=True)
        polymer_slab.sld.real.setp(bounds=(0.3, 6.0), vary=True)
        polymer_slab.rough.setp(bounds=(1, 25), vary=True)
        sio2_slab.thick.setp(bounds=(3, 50), vary=True)
        sio2_slab.rough.setp(bounds=(0.5, 15), vary=True)
        si_slab.rough.setp(bounds=(0.5, 15), vary=True)

    structure = air | polymer_slab | sio2_slab | si_slab
    model = ReflectModel(structure, bkg=params["bkg"], dq=dq_over_q * 100)

    if vary:
        model.bkg.setp(bounds=(1e-10, 1e-5), vary=True)

    return structure, model, {
        "polymer_slab": polymer_slab,
        "sio2_slab": sio2_slab,
        "si_slab": si_slab,
    }

def load_and_preprocess_data(gt_params, q_min, q_max, n_points, noise_frac, seed, dq_over_q):
    """
    Generate synthetic XRR benchmark data.

    1. Build ground-truth structure with refnx.
    2. Compute clean reflectivity R(Q) via refnx forward.
    3. Add realistic Poisson-like noise + background.

    Parameters
    ----------
    gt_params : dict
        Ground-truth layer parameters.
    q_min : float
        Minimum Q value [Å^-1].
    q_max : float
        Maximum Q value [Å^-1].
    n_points : int
        Number of Q points.
    noise_frac : float
        Relative noise level.
    seed : int
        Random seed for reproducibility.
    dq_over_q : float
        Resolution smearing parameter.

    Returns
    -------
    data_dict : dict
        Contains q, R_meas, R_clean, R_err, gt_params, dq_over_q
    """
    print("[DATA] Building ground-truth multilayer with refnx ...")
    q = np.linspace(q_min, q_max, n_points)

    # Compute clean reflectivity using forward operator
    _, model, _ = build_structure(gt_params, vary=False, dq_over_q=dq_over_q)
    R_clean = model(q)
    print(f"[DATA] R(Q) range: [{R_clean.min():.3e}, {R_clean.max():.3e}]")

    # Realistic noise: relative Gaussian noise scaled by sqrt(R)
    rng = np.random.default_rng(seed)
    sigma_R = np.maximum(R_clean * noise_frac, 1e-12)
    R_noisy = R_clean + sigma_R * rng.standard_normal(n_points)
    R_noisy = np.maximum(R_noisy, 1e-12)

    R_err = sigma_R

    print(f"[DATA] Added {noise_frac*100:.0f}% relative noise "
          f"({n_points} points, Q ∈ [{q_min}, {q_max}] Å⁻¹)")

    data_dict = {
        "q": q,
        "R_meas": R_noisy,
        "R_clean": R_clean,
        "R_err": R_err,
        "gt_params": gt_params,
        "dq_over_q": dq_over_q,
    }

    return data_dict
