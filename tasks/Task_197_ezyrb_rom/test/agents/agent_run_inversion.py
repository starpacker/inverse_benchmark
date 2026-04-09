import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

import time

from ezyrb import POD, RBF, Database, ReducedOrderModel

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def run_inversion(params_train, snapshots_train, k_test_values, nx, ny):
    """
    Build and run the Reduced-Order Model inversion using POD + RBF.
    
    Parameters
    ----------
    params_train : ndarray of shape (n_train, 1)
        Training parameter values.
    snapshots_train : ndarray of shape (n_train, nx*ny)
        Training snapshot data.
    k_test_values : array-like
        Test parameter values for prediction.
    nx, ny : int
        Grid resolution.
    
    Returns
    -------
    predictions : ndarray of shape (n_test, nx*ny)
        Predicted temperature fields at test parameters.
    rom_info : dict
        Information about the ROM (n_modes, fit_time, etc.)
    """
    db = Database(params_train, snapshots_train)
    pod = POD('svd')
    rbf = RBF()
    rom = ReducedOrderModel(db, pod, rbf)
    
    t0 = time.time()
    rom.fit()
    fit_time = time.time() - t0
    
    n_modes = rom.reduction.singular_values.shape[0]
    
    predictions = []
    for k_val in k_test_values:
        pred = rom.predict([k_val]).flatten()
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    rom_info = {
        'n_modes': int(n_modes),
        'fit_time': fit_time,
        'n_train': len(params_train),
        'rom': rom
    }
    
    return predictions, rom_info
