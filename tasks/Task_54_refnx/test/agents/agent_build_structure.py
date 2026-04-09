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
