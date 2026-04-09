import matplotlib

matplotlib.use('Agg')

import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_fsc_aspire(gt_data, recon_data):
    """Compute Fourier Shell Correlation using ASPIRE."""
    from aspire.volume import Volume

    gt_vol = Volume(gt_data[np.newaxis, ...].astype(np.float64))
    recon_vol = Volume(recon_data[np.newaxis, ...].astype(np.float64))

    try:
        est_res, fsc_curve = gt_vol.fsc(recon_vol, cutoff=0.5)
        valid = fsc_curve[1:len(fsc_curve)//2]
        valid = valid[np.isfinite(valid)]
        mean_fsc = float(np.mean(valid)) if len(valid) > 0 else 0.0
        return mean_fsc, fsc_curve
    except Exception as e:
        print(f"  FSC computation failed: {e}")
        return None, None
