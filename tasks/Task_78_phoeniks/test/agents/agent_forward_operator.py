import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

import sys

from scipy.constants import c as c_0

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")

if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

def forward_operator(n, k, frequency, thickness):
    """
    Compute the transfer function H(ω) for a single dielectric layer.
    
    The forward model describes THz pulse propagation through a dielectric slab:
    H(ω) = t12 * t21 * P(ω) / P_air(ω) * FP(ω)
    
    where:
      - t12, t21: Fresnel transmission coefficients at air-material interfaces
      - P(ω): propagation phase through material
      - P_air(ω): propagation phase through air (same thickness)
      - FP(ω): Fabry-Perot etalon factor for multiple reflections
      
    Physics:
      - Complex refractive index: ñ = n - i*k
      - Propagation: exp(-i*ω*ñ*d/c)
      - Fresnel: t = 2n1/(n1+n2), r = (n2-n1)/(n1+n2)
      
    Parameters:
        n: refractive index array (per frequency)
        k: extinction coefficient array (per frequency)
        frequency: frequency array (Hz)
        thickness: sample thickness (m)
    
    Returns:
        H: complex transfer function array
    """
    omega = 2 * np.pi * frequency
    complex_n = n - 1j * k  # complex refractive index
    
    # Fresnel coefficients (air -> material -> air)
    # n_air = 1
    t12 = 2.0 / (1.0 + complex_n)           # air to material
    t21 = 2.0 * complex_n / (1.0 + complex_n)  # material to air
    r22 = (complex_n - 1.0) / (1.0 + complex_n)  # reflection at material-air interface
    
    # Propagation through material
    propagation = np.exp(-1j * omega * complex_n * thickness / c_0)
    
    # Propagation through air (reference path)
    propagation_air = np.exp(-1j * omega * thickness / c_0)
    
    # Fabry-Perot factor (multiple reflections inside slab)
    rr = r22 * r22
    FP = 1.0 / (1.0 - rr * (propagation ** 2))
    
    # Total transfer function
    H = t12 * t21 * propagation * FP / propagation_air
    
    return H
