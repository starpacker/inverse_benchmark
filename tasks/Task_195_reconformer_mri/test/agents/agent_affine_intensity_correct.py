import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def affine_intensity_correct(recon, gt):
    """Optimal affine intensity correction: recon_corrected = a * recon + b."""
    r = recon.flatten()
    g = gt.flatten()
    N = len(r)
    
    sr2 = np.dot(r, r)
    sr = r.sum()
    srg = np.dot(r, g)
    sg = g.sum()
    
    det = sr2 * N - sr * sr
    if abs(det) < 1e-12:
        return recon
    
    a = (srg * N - sr * sg) / det
    b = (sr2 * sg - sr * srg) / det
    
    corrected = a * recon + b
    print(f"  Intensity correction: scale={a:.4f}, offset={b:.4f}")
    return corrected
