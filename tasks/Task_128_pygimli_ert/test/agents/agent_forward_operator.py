import matplotlib

matplotlib.use('Agg')

import os

from pygimli.physics import ert

def forward_operator(preprocess_data):
    """
    Forward simulation: generate synthetic apparent resistivity data.
    
    Parameters
    ----------
    preprocess_data : dict
        Output from load_and_preprocess_data
    
    Returns
    -------
    dict containing:
        - data: simulated ERT data with noise
        - data_file: path to saved data file
        - all keys from preprocess_data
    """
    print("[3/6] Running forward simulation (adding 3% noise + 1µV)...")

    mesh = preprocess_data['mesh']
    scheme = preprocess_data['scheme']
    rhomap = preprocess_data['rhomap']
    results_dir = preprocess_data['results_dir']

    data = ert.simulate(mesh, scheme=scheme, res=rhomap,
                        noiseLevel=0.03, noiseAbs=1e-6, seed=42)

    n_before = data.size()
    data.remove(data['rhoa'] < 0)
    n_after = data.size()
    if n_before != n_after:
        print(f"   Removed {n_before - n_after} negative rhoa values")

    print(f"   Simulated data points: {data.size()}")
    print(f"   Apparent resistivity range: [{min(data['rhoa']):.2f}, {max(data['rhoa']):.2f}] Ohm·m")

    data_file = os.path.join(results_dir, 'ert_data.dat')
    data.save(data_file)

    result = preprocess_data.copy()
    result['data'] = data
    result['data_file'] = data_file
    return result
