import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

from scipy.optimize import least_squares

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

ASSETS_DIR = "/data/yjh/website_assets/Task_113_diffpy_pdf"

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(ASSETS_DIR, exist_ok=True)

def fcc_neighbor_distances(a, r_max, max_shell=200):
    """
    Compute interatomic distances and coordination numbers for FCC structure.

    For FCC, neighbor distances are a × sqrt(n/2) for certain n values.
    Returns list of (distance, coordination_number) pairs.
    """
    distances = []
    n_max = int(np.ceil(r_max / a)) + 1
    for h in range(-n_max, n_max + 1):
        for k in range(-n_max, n_max + 1):
            for l in range(-n_max, n_max + 1):
                for bx, by, bz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    x = (h + bx) * a
                    y = (k + by) * a
                    z = (l + bz) * a
                    d = np.sqrt(x**2 + y**2 + z**2)
                    if 0.1 < d < r_max:
                        distances.append(d)

    distances = np.sort(distances)
    shells = []
    tol = 0.01
    i = 0
    while i < len(distances) and len(shells) < max_shell:
        d_ref = distances[i]
        count = 0
        while i < len(distances) and abs(distances[i] - d_ref) < tol:
            count += 1
            i += 1
        shells.append((d_ref, count))

    return shells

def forward_operator(r, a, B, scale, r_max):
    """
    Compute the reduced pair distribution function G(r) for FCC structure.
    
    G(r) = scale × Σ_n C_n × exp(-(r - d_n)² / (2σ_n²)) / (σ_n√(2π) × r)
    
    where σ_n² = B (isotropic Debye-Waller), C_n = coordination number
    
    Parameters:
    -----------
    r : ndarray
        r grid values in Angstroms
    a : float
        Lattice constant in Angstroms
    B : float
        Debye-Waller factor in Angstroms squared
    scale : float
        Overall scale factor
    r_max : float
        Maximum r value for computing neighbor shells
    
    Returns:
    --------
    G : ndarray
        Computed G(r) values
    """
    shells = fcc_neighbor_distances(a, r_max)
    sigma = np.sqrt(B)
    
    G = np.zeros_like(r)
    rho0 = 4 / a**3
    
    for d_n, coord_n in shells:
        sigma_n = sigma * np.sqrt(1 + 0.002 * d_n**2)
        amplitude = coord_n / (4 * np.pi * d_n**2 * rho0)
        peak = amplitude * np.exp(-0.5 * ((r - d_n) / sigma_n)**2) / (sigma_n * np.sqrt(2 * np.pi))
        G += peak
    
    G = scale * G / (np.max(np.abs(G)) + 1e-12)
    G *= np.exp(-0.01 * r**2)
    
    return G

def run_inversion(r, G_measured, r_max):
    """
    Fit structural parameters from measured G(r) using least-squares optimization.
    
    Parameters to fit:
      - a: lattice constant
      - B: Debye-Waller factor
      - scale: overall scale factor
    
    Parameters:
    -----------
    r : ndarray
        r grid values
    G_measured : ndarray
        Measured (noisy) G(r) values
    r_max : float
        Maximum r value for forward model
    
    Returns:
    --------
    a_fit : float
        Fitted lattice constant
    B_fit : float
        Fitted Debye-Waller factor
    scale_fit : float
        Fitted scale factor
    G_fit : ndarray
        G(r) computed with fitted parameters
    optimization_result : OptimizeResult
        Full scipy optimization result object
    """
    def residual_func(params):
        a, B, scale = params
        if a < 2.0 or a > 6.0 or B < 0.01 or B > 2.0 or scale < 0.1 or scale > 5.0:
            return np.ones_like(r) * 1e6
        G_model = forward_operator(r, a, B, scale, r_max)
        return G_model - G_measured
    
    a0 = 3.5
    B0 = 0.4
    scale0 = 0.8
    
    result = least_squares(
        residual_func,
        x0=[a0, B0, scale0],
        bounds=([2.0, 0.01, 0.1], [6.0, 2.0, 5.0]),
        method="trf",
        max_nfev=2000,
    )
    
    a_fit, B_fit, scale_fit = result.x
    G_fit = forward_operator(r, a_fit, B_fit, scale_fit, r_max)
    
    return a_fit, B_fit, scale_fit, G_fit, result
