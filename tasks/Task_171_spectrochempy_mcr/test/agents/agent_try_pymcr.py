import matplotlib

matplotlib.use('Agg')

import numpy as np

import os

np.random.seed(42)

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

def try_pymcr(D, n_components):
    """Try using pyMCR for MCR-ALS decomposition."""
    try:
        from pymcr.mcr import McrAR
        from pymcr.constraints import ConstraintNonneg, ConstraintNorm
        from pymcr.regressors import NNLS

        # SVD-based initial guess
        U, s, Vt = np.linalg.svd(D, full_matrices=False)
        S_init = np.abs(Vt[:n_components, :])

        mcr = McrAR(
            max_iter=500,
            tol_increase=50,
            tol_n_increase=20,
            tol_err_change=1e-8,
            c_regr=NNLS(),
            st_regr=NNLS(),
            c_constraints=[ConstraintNonneg()],
            st_constraints=[ConstraintNonneg(), ConstraintNorm()]
        )
        mcr.fit(D, ST=S_init)
        C_recovered = mcr.C_opt_
        S_recovered = mcr.ST_opt_
        print(f"pyMCR converged. Final error: {mcr.err_[-1]:.6f}")
        return C_recovered, S_recovered, True
    except Exception as e:
        print(f"pyMCR failed: {e}")
        return None, None, False
