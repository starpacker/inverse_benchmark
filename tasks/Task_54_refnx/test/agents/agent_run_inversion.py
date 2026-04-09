import matplotlib

matplotlib.use('Agg')

from refnx.reflect import SLD as SLDobj, ReflectModel, Structure

from refnx.dataset import ReflectDataset

from refnx.analysis import Objective, CurveFitter

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

def run_inversion(data_dict, seed=42):
    """
    Fit the XRR curve using refnx's Differential Evolution +
    Levenberg-Marquardt pipeline.

    The refnx CurveFitter wraps scipy.optimize.differential_evolution
    for global search, then scipy.optimize.least_squares for local
    refinement. Resolution smearing (dq) is handled internally.

    Parameters
    ----------
    data_dict : dict
        Contains q, R_meas, R_err, gt_params, dq_over_q.
    seed : int
        Random seed for DE optimization.

    Returns
    -------
    result_dict : dict
        Contains fitted_params, R_fit, chi2, q, R_meas, R_clean, R_err, gt_params.
    """
    q = data_dict["q"]
    R_meas = data_dict["R_meas"]
    R_err = data_dict["R_err"]
    R_clean = data_dict["R_clean"]
    gt_params = data_dict["gt_params"]
    dq_over_q = data_dict["dq_over_q"]

    # Build fitting model with free parameters
    initial_guess = {
        "polymer_thick": 200.0,
        "polymer_sld": 2.0,
        "polymer_rough": 5.0,
        "sio2_thick": 10.0,
        "sio2_sld": gt_params["sio2_sld"],
        "sio2_rough": 5.0,
        "si_sld": gt_params["si_sld"],
        "si_rough": 5.0,
        "bkg": 1e-7,
    }

    structure, model, slabs = build_structure(initial_guess, vary=True, dq_over_q=dq_over_q)

    # Create refnx dataset (Q, R, dR, dQ)
    dq = q * dq_over_q
    dataset = ReflectDataset(data=(q, R_meas, R_err, dq))

    objective = Objective(model, dataset)

    # Stage 1: Differential Evolution (global)
    print("[RECON] Stage 1 — Differential Evolution (refnx.CurveFitter) ...")
    fitter = CurveFitter(objective)
    fitter.fit('differential_evolution', seed=seed, maxiter=150, tol=1e-5)
    chi2_de = objective.chisqr()
    print(f"[RECON]   χ² after DE = {chi2_de:.4f}")

    # Stage 2: Least-Squares refinement (local)
    print("[RECON] Stage 2 — Least-Squares refinement ...")
    fitter.fit('least_squares')
    chi2_lm = objective.chisqr()
    print(f"[RECON]   χ² after LM = {chi2_lm:.4f}")

    # Extract fitted parameter values
    p = slabs["polymer_slab"]
    s = slabs["sio2_slab"]
    si = slabs["si_slab"]

    fitted_params = {
        "polymer_thick": float(p.thick.value),
        "polymer_sld": float(p.sld.real.value),
        "polymer_rough": float(p.rough.value),
        "sio2_thick": float(s.thick.value),
        "sio2_sld": float(s.sld.real.value),
        "sio2_rough": float(s.rough.value),
        "si_sld": gt_params["si_sld"],
        "si_rough": float(si.rough.value),
        "bkg": float(model.bkg.value),
    }

    R_fit = model(q, x_err=dq)

    result_dict = {
        "fitted_params": fitted_params,
        "R_fit": R_fit,
        "chi2": chi2_lm,
        "q": q,
        "R_meas": R_meas,
        "R_clean": R_clean,
        "R_err": R_err,
        "gt_params": gt_params,
    }

    return result_dict
