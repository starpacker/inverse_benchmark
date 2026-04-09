import matplotlib

matplotlib.use('Agg')

import os

import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

REPO_DIR = os.path.join(SCRIPT_DIR, "repo")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

sys.path.insert(0, REPO_DIR)

os.makedirs(RESULTS_DIR, exist_ok=True)

from pybaselines import Baseline

def run_inversion(x, measured, true_baseline):
    """
    Run multiple baseline estimation algorithms and select the best one.
    
    This is the inverse problem: given the measured spectrum, estimate
    the baseline component using various algorithms from pybaselines.
    The best algorithm is selected based on RMSE to ground truth.
    
    Parameters
    ----------
    x : np.ndarray
        Wavenumber axis.
    measured : np.ndarray
        Measured spectrum.
    true_baseline : np.ndarray
        Ground truth baseline (for algorithm selection).
    
    Returns
    -------
    result_dict : dict
        Dictionary containing:
        - 'best_name': name of the best algorithm
        - 'est_baseline': estimated baseline from best algorithm
        - 'best_rmse': RMSE of best algorithm
        - 'all_results': dict of all algorithm results
    """
    print("[RECON] Running baseline estimation algorithms...")
    baseline_fitter = Baseline(x_data=x)
    results = {}

    # Algorithm 1: AsLS (Asymmetric Least Squares)
    print("[RECON]   Running AsLS (Asymmetric Least Squares)...")
    try:
        bline_asls, params_asls = baseline_fitter.asls(measured, lam=1e7, p=0.01)
        results['AsLS'] = bline_asls
        print(f"[RECON]   AsLS done. tol_history length: {len(params_asls.get('tol_history', []))}")
    except Exception as e:
        print(f"[RECON]   AsLS failed: {e}")

    # Algorithm 2: airPLS (Adaptive Iteratively Reweighted Penalized Least Squares)
    print("[RECON]   Running airPLS...")
    try:
        bline_airpls, params_airpls = baseline_fitter.airpls(measured, lam=1e7)
        results['airPLS'] = bline_airpls
        print(f"[RECON]   airPLS done. tol_history length: {len(params_airpls.get('tol_history', []))}")
    except Exception as e:
        print(f"[RECON]   airPLS failed: {e}")

    # Algorithm 3: SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping)
    print("[RECON]   Running SNIP...")
    try:
        bline_snip, params_snip = baseline_fitter.snip(
            measured, max_half_window=80, decreasing=True, smooth_half_window=3
        )
        results['SNIP'] = bline_snip
        print(f"[RECON]   SNIP done.")
    except Exception as e:
        print(f"[RECON]   SNIP failed: {e}")

    # Algorithm 4: iarpls (Improved Asymmetrically Reweighted Penalized Least Squares)
    print("[RECON]   Running IarPLS...")
    try:
        bline_iarpls, params_iarpls = baseline_fitter.iarpls(measured, lam=1e7)
        results['IarPLS'] = bline_iarpls
        print(f"[RECON]   IarPLS done.")
    except Exception as e:
        print(f"[RECON]   IarPLS failed: {e}")

    # Algorithm 5: ModPoly (Modified Polynomial)
    print("[RECON]   Running ModPoly...")
    try:
        bline_modpoly, params_modpoly = baseline_fitter.modpoly(measured, poly_order=5)
        results['ModPoly'] = bline_modpoly
        print(f"[RECON]   ModPoly done.")
    except Exception as e:
        print(f"[RECON]   ModPoly failed: {e}")

    print(f"[RECON] Completed {len(results)} algorithms successfully.")

    # Select best algorithm based on RMSE to ground truth baseline
    print("[RECON] Selecting best algorithm...")
    best_name = None
    best_rmse = np.inf
    best_baseline = None

    for name, est_baseline in results.items():
        rmse = np.sqrt(np.mean((est_baseline - true_baseline) ** 2))
        print(f"[RECON]   {name:10s} baseline RMSE = {rmse:.6f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_baseline = est_baseline

    print(f"[RECON] Best algorithm: {best_name} (RMSE={best_rmse:.6f})")
    
    result_dict = {
        'best_name': best_name,
        'est_baseline': best_baseline,
        'best_rmse': best_rmse,
        'all_results': results
    }
    
    return result_dict
