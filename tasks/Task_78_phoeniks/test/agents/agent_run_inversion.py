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

def run_inversion(data_dict, thickness, freq_start, freq_stop):
    """
    Extract n(ω) and κ(ω) from THz-TDS data.
    
    Algorithm:
      1. Extract transfer function from preprocessed data
      2. Unwrap phase of transfer function in specified frequency range
      3. Get initial estimates of n, k from analytical approximation
      4. Optimize n, k at each frequency by minimizing error between
         measured and modeled H(ω)
    
    The analytical initial estimate comes from:
      n ≈ 1 + c * φ / (ω * d)
      k ≈ -c * ln(|H|) / (ω * d)  (simplified, neglecting Fresnel losses)
    
    The optimization minimizes |H_model(n,k) - H_measured|² at each frequency.
    
    Parameters:
        data_dict: output from load_and_preprocess_data
        thickness: sample thickness (m)
        freq_start: start frequency for extraction (Hz)
        freq_stop: stop frequency for extraction (Hz)
    
    Returns:
        dict with 'frequency', 'n', 'k', 'alpha' arrays
    """
    frequency = data_dict['frequency']
    transfer_function = data_dict['transfer_function']
    
    # Select frequency range
    freq_mask = (frequency >= freq_start) & (frequency <= freq_stop)
    freq_selected = frequency[freq_mask]
    H_measured = transfer_function[freq_mask]
    
    n_freq = len(freq_selected)
    
    # Unwrap phase
    phase = np.unwrap(np.angle(H_measured))
    magnitude = np.abs(H_measured)
    
    # Initial estimates from analytical approximation
    omega = 2 * np.pi * freq_selected
    
    # From phase: n ≈ 1 - c * φ / (ω * d)
    # The negative sign comes from the propagation convention exp(-iωnd/c)
    n_init = 1.0 - c_0 * phase / (omega * thickness)
    
    # From magnitude: simplified estimate
    # |H| ≈ |t12*t21| * exp(-k*ω*d/c)
    # For n ≈ 1.5: |t12*t21| ≈ 0.96
    # So: k ≈ -c * ln(|H|/0.96) / (ω * d)
    t_factor = 4.0 * n_init / ((1 + n_init) ** 2)
    t_factor = np.clip(t_factor, 0.1, 1.0)
    k_init = -c_0 * np.log(magnitude / t_factor + 1e-10) / (omega * thickness)
    k_init = np.clip(k_init, 0, 1.0)
    
    # Optimization at each frequency point
    n_opt = np.zeros(n_freq)
    k_opt = np.zeros(n_freq)
    
    from scipy.optimize import minimize
    
    for i in range(n_freq):
        freq_i = freq_selected[i:i+1]
        H_meas_i = H_measured[i]
        
        def objective(params):
            n_val, k_val = params
            H_model = forward_operator(
                np.array([n_val]), 
                np.array([k_val]), 
                freq_i, 
                thickness
            )[0]
            # Cost: squared error in real and imaginary parts
            error = np.abs(H_model - H_meas_i) ** 2
            return error
        
        # Initial guess
        x0 = [n_init[i], max(k_init[i], 0.001)]
        
        # Bounds: n > 1 (typical for dielectrics), k >= 0
        bounds = [(1.0, 5.0), (0.0, 1.0)]
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 100, 'ftol': 1e-12})
            n_opt[i] = result.x[0]
            k_opt[i] = max(result.x[1], 0)
        except Exception:
            n_opt[i] = n_init[i]
            k_opt[i] = max(k_init[i], 0)
    
    # Compute absorption coefficient: α = 2ωk/c
    alpha_opt = 2 * omega * k_opt / c_0
    
    return {
        'frequency': freq_selected,
        'n': n_opt,
        'k': k_opt,
        'alpha': alpha_opt,
    }
