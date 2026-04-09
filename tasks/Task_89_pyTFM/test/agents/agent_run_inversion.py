import matplotlib

matplotlib.use('Agg')

import os

import sys

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")

if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

from pyTFM.TFM_functions import ffttc_traction

def run_inversion(measurements, young, sigma, pixelsize1, pixelsize2):
    """
    FTTC inverse solver: recover traction forces from displacement field.

    Uses pyTFM's Fourier Transform Traction Cytometry with Gaussian filtering
    for regularization (noise suppression in high-frequency components).

    Parameters:
        measurements: tuple (u, v) displacement fields in pixels
        young: float, Young's modulus in Pa
        sigma: float, Poisson's ratio
        pixelsize1: float, pixel size in meters (image pixel size)
        pixelsize2: float, pixel size in meters (deformation field pixel size)

    Returns:
        result: tuple (tx_recon, ty_recon) reconstructed traction fields in Pa
    """
    u, v = measurements

    print("[RECON] Running FTTC inverse solver...")
    
    # Use pyTFM's FTTC solver
    # spatial_filter="gaussian" with fs for regularization
    tx_recon, ty_recon = ffttc_traction(
        u, v,
        pixelsize1=pixelsize1,
        pixelsize2=pixelsize2,
        young=young,
        sigma=sigma,
        spatial_filter="gaussian",
        fs=3  # smaller filter for less smoothing (in auto-units)
    )

    print(f"[RECON] Reconstructed traction field shape: {tx_recon.shape}")
    
    return (tx_recon, ty_recon)
