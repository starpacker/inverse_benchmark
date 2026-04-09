import matplotlib

matplotlib.use('Agg')

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def reconstruct_mean_estimator(sim):
    """
    Reconstruct volume using ASPIRE's MeanEstimator.
    """
    from aspire.reconstruction import MeanEstimator

    print("  Running ASPIRE MeanEstimator...")
    estimator = MeanEstimator(sim)
    recon_vol = estimator.estimate()
    recon_data = recon_vol.asnumpy().squeeze()
    print(f"  MeanEstimator reconstruction shape: {recon_data.shape}")
    return recon_data
