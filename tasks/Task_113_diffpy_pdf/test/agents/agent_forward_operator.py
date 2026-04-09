import numpy as np

import matplotlib

matplotlib.use("Agg")

import os

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
