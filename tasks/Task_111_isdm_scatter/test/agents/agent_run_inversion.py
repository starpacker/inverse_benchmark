import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, binary_fill_holes

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR  = "/data/yjh/website_assets/Task_111_isdm_scatter"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def shrinkwrap_update(g, support, sigma=2.0, threshold_fraction=0.1):
    """
    Shrinkwrap support update: blur the current estimate and threshold.
    """
    blurred = gaussian_filter(np.abs(g), sigma=sigma)
    threshold = threshold_fraction * np.max(blurred)
    new_support = blurred > threshold
    return new_support

def compute_r_factor(measured_magnitude, recon):
    """Compute R-factor (Fourier-space error) to evaluate reconstruction quality."""
    F_recon = np.fft.fft2(recon)
    recon_mag = np.abs(F_recon)
    r_factor = np.sum(np.abs(measured_magnitude - recon_mag)) / np.sum(measured_magnitude + 1e-12)
    return r_factor

def run_inversion(measured_magnitude, support, n_iter, beta, n_restarts=10):
    """
    Run HIO phase retrieval with multiple random restarts to recover the object.
    
    Given |F[obj]| (Fourier magnitude), recover the object by iterating
    between Fourier and real-space constraints using the Hybrid Input-Output
    (HIO) algorithm with shrinkwrap support updates.
    
    Args:
        measured_magnitude: |F[obj]| measured from speckle autocorrelation
        support: Binary mask defining the initial support constraint
        n_iter: Number of HIO iterations per restart
        beta: HIO feedback parameter
        n_restarts: Number of random restarts for phase retrieval
    
    Returns:
        best_recon: Best reconstruction (lowest R-factor)
    """
    best_recon = None
    best_r_factor = np.inf
    n = measured_magnitude.shape[0]

    for restart_idx in range(n_restarts):
        # Initialize with random phase
        phase = 2 * np.pi * np.random.rand(n, n)
        g = np.real(np.fft.ifft2(measured_magnitude * np.exp(1j * phase)))
        
        # Copy support for this restart
        current_support = support.copy()

        for iteration in range(n_iter):
            # Fourier constraint: replace magnitude, keep phase
            G = np.fft.fft2(g)
            G_constrained = measured_magnitude * np.exp(1j * np.angle(G))
            g_prime = np.real(np.fft.ifft2(G_constrained))

            # Real-space constraint: HIO update (vectorized)
            valid = current_support & (g_prime >= 0)
            g_new = np.where(valid, g_prime, g - beta * g_prime)
            g = g_new

            # Every 50 iterations, apply Error Reduction (ER) step
            if (iteration + 1) % 50 == 0:
                G = np.fft.fft2(g)
                G_constrained = measured_magnitude * np.exp(1j * np.angle(G))
                g_prime = np.real(np.fft.ifft2(G_constrained))
                g = g_prime * current_support
                g[g < 0] = 0

            # Shrinkwrap: update support every 100 iterations after initial 200
            if iteration >= 200 and (iteration + 1) % 100 == 0:
                sigma = max(1.0, 3.0 - iteration / 1000.0)  # decreasing sigma
                current_support = shrinkwrap_update(g, current_support, sigma=sigma, threshold_fraction=0.08)

        # Final cleanup
        g = g * current_support
        g[g < 0] = 0
        
        # Compute R-factor for this restart
        r_fac = compute_r_factor(measured_magnitude, g)
        print(f"    Restart {restart_idx+1}/{n_restarts}: R-factor = {r_fac:.6f}")
        
        if r_fac < best_r_factor:
            best_r_factor = r_fac
            best_recon = g.copy()

    print(f"  Best R-factor: {best_r_factor:.6f}")
    return best_recon
