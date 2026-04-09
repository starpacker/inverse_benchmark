import matplotlib

matplotlib.use("Agg")

import numpy as np

from phasorpy.components import two_fractions_from_phasor

from phasorpy.phasor import (
    phasor_from_lifetime,
    phasor_from_signal,
    phasor_semicircle,
)

def run_inversion(
    noisy_signal: np.ndarray,
    freq_mhz: float,
    tau1_ns: float,
    tau2_ns: float,
) -> dict:
    """
    Inverse problem: Extract fluorescence lifetime component fractions.
    
    Performs phasor analysis on the time-domain FLIM data:
    1. Compute phasor coordinates (G, S) from the signal using FFT
    2. Compute reference phasor positions for known lifetime components
    3. Apply lever rule (linear unmixing) to decompose each pixel's phasor
    
    Parameters
    ----------
    noisy_signal : np.ndarray
        Noisy FLIM data, shape (nx, ny, n_time).
    freq_mhz : float
        Laser repetition frequency in MHz.
    tau1_ns : float
        Lifetime of species 1 in nanoseconds.
    tau2_ns : float
        Lifetime of species 2 in nanoseconds.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'f1_recon': Reconstructed fraction map for species 1 (nx, ny)
        - 'G_meas': Measured G (real) phasor coordinates (nx, ny)
        - 'S_meas': Measured S (imaginary) phasor coordinates (nx, ny)
        - 'G_ref': Reference G coordinates for the two components (2,)
        - 'S_ref': Reference S coordinates for the two components (2,)
        - 'mean_intensity': Mean intensity at each pixel (nx, ny)
    """
    # Phasor transform: compute (G, S) for each pixel
    # phasor_from_signal expects signal along axis; returns (mean, real, imag)
    mean_intensity, G_meas, S_meas = phasor_from_signal(
        noisy_signal, axis=-1, harmonic=1
    )
    
    print(f"Phasor coords: G range [{G_meas.min():.4f}, {G_meas.max():.4f}], "
          f"S range [{S_meas.min():.4f}, {S_meas.max():.4f}]")
    
    # Compute known component phasor positions on the semicircle
    G_ref, S_ref = phasor_from_lifetime(freq_mhz, np.array([tau1_ns, tau2_ns]))
    
    print(f"Component phasors: τ₁={tau1_ns} ns → G={G_ref[0]:.4f}, S={S_ref[0]:.4f}")
    print(f"                   τ₂={tau2_ns} ns → G={G_ref[1]:.4f}, S={S_ref[1]:.4f}")
    
    # Inverse: linear unmixing via lever rule
    f1_recon = two_fractions_from_phasor(
        G_meas, S_meas,
        G_ref, S_ref,
    )
    f1_recon = np.clip(f1_recon, 0.0, 1.0)
    
    print(f"Recovered f1 range [{f1_recon.min():.4f}, {f1_recon.max():.4f}]")
    
    return {
        'f1_recon': f1_recon,
        'G_meas': G_meas,
        'S_meas': S_meas,
        'G_ref': G_ref,
        'S_ref': S_ref,
        'mean_intensity': mean_intensity,
    }
