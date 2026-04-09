import matplotlib

matplotlib.use('Agg')

from pygimli.physics import ert

def run_inversion(forward_data):
    """
    Inversion: Gauss-Newton with smoothness regularization.
    
    Parameters
    ----------
    forward_data : dict
        Output from forward_operator
    
    Returns
    -------
    dict containing:
        - inv_model: inverted model
        - inv_model_pd: model on para domain
        - pd: para domain mesh
        - mgr: ERT manager
        - chi2: chi-squared value
        - all keys from forward_data
    """
    print("[4/6] Running ERT inversion (Gauss-Newton, lambda=1)...")

    data = forward_data['data']

    mgr = ert.ERTManager(data)

    inv_model = mgr.invert(lam=1, zWeight=0.3, verbose=True)

    chi2 = mgr.inv.chi2()
    print(f"   Inversion chi² = {chi2:.3f}")
    print(f"   Inversion model size: {len(inv_model)}")

    pd = mgr.paraDomain
    inv_model_pd = mgr.paraModel(inv_model)

    print(f"   Para domain: {pd.cellCount()} cells")
    print(f"   Inverted resistivity range: [{min(inv_model_pd):.2f}, {max(inv_model_pd):.2f}] Ohm·m")

    result = forward_data.copy()
    result['inv_model'] = inv_model
    result['inv_model_pd'] = inv_model_pd
    result['pd'] = pd
    result['mgr'] = mgr
    result['chi2'] = chi2
    return result
