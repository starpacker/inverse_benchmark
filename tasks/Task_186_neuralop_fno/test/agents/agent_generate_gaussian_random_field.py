import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

import sys

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")

if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_gaussian_random_field(n, resolution, alpha=2.0, tau=3.0, seed=None):
    """Generate Gaussian Random Field for permeability coefficients."""
    if seed is not None:
        np.random.seed(seed)
    
    fields = []
    for _ in range(n):
        k1 = np.fft.fftfreq(resolution, d=1.0/resolution)
        k2 = np.fft.fftfreq(resolution, d=1.0/resolution)
        K1, K2 = np.meshgrid(k1, k2)
        
        power = (tau**2 + K1**2 + K2**2)**(-alpha/2.0)
        power[0, 0] = 0
        
        coeff_real = np.random.randn(resolution, resolution)
        coeff_imag = np.random.randn(resolution, resolution)
        coeff = (coeff_real + 1j * coeff_imag) * power
        
        field = np.real(np.fft.ifft2(coeff * resolution))
        field = np.exp(field)
        field = 3 + 9 * (field - field.min()) / (field.max() - field.min() + 1e-8)
        
        fields.append(field)
    
    return np.array(fields, dtype=np.float32)
