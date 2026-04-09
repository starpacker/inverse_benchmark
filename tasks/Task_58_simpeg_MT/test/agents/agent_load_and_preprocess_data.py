import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

GT_THICKNESSES = [200.0, 500.0]

GT_RESISTIVITIES = [100.0, 10.0, 1000.0]

N_FREQ = 50

FREQ_MIN = 0.001

FREQ_MAX = 100.0

NOISE_FLOOR = 0.01

NOISE_PCT = 0.02

SEED = 42

def load_and_preprocess_data():
    """
    Generate synthetic 1D MT sounding data.
    
    Returns
    -------
    data_dict : dict
        Dictionary containing:
        - frequencies: array of frequencies [Hz]
        - Z_clean: clean complex impedance
        - Z_noisy: noisy complex impedance
        - rho_clean: clean apparent resistivity [Ohm·m]
        - rho_noisy: noisy apparent resistivity [Ohm·m]
        - phi_clean: clean phase [degrees]
        - phi_noisy: noisy phase [degrees]
        - gt_thicknesses: ground truth layer thicknesses
        - gt_resistivities: ground truth layer resistivities
    """
    print("[DATA] Generating synthetic 1D MT data ...")
    
    mu0 = 4 * np.pi * 1e-7  # H/m
    
    frequencies = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_FREQ)
    
    # Compute clean impedance using Wait's recursion
    Z_clean, rho_clean, phi_clean = forward_operator(
        frequencies, GT_THICKNESSES, GT_RESISTIVITIES
    )
    
    print(f"[DATA] {N_FREQ} frequencies: [{FREQ_MIN:.4f}, {FREQ_MAX:.1f}] Hz")
    print(f"[DATA] ρ_a range: [{rho_clean.min():.1f}, {rho_clean.max():.1f}] Ω·m")
    print(f"[DATA] φ range: [{phi_clean.min():.1f}, {phi_clean.max():.1f}]°")
    
    # Add noise to impedance
    rng = np.random.default_rng(SEED)
    noise_real = (NOISE_FLOOR * np.abs(Z_clean) +
                  NOISE_PCT * np.abs(Z_clean)) * rng.standard_normal(N_FREQ)
    noise_imag = (NOISE_FLOOR * np.abs(Z_clean) +
                  NOISE_PCT * np.abs(Z_clean)) * rng.standard_normal(N_FREQ)
    Z_noisy = Z_clean + noise_real + 1j * noise_imag
    
    rho_noisy = np.abs(Z_noisy) ** 2 / (2 * np.pi * frequencies * mu0)
    phi_noisy = np.degrees(np.angle(Z_noisy))
    
    data_dict = {
        'frequencies': frequencies,
        'Z_clean': Z_clean,
        'Z_noisy': Z_noisy,
        'rho_clean': rho_clean,
        'rho_noisy': rho_noisy,
        'phi_clean': phi_clean,
        'phi_noisy': phi_noisy,
        'gt_thicknesses': GT_THICKNESSES,
        'gt_resistivities': GT_RESISTIVITIES,
    }
    
    return data_dict

def forward_operator(frequencies, thicknesses, resistivities):
    """
    Compute 1D MT impedance using Wait's recursive formula.

    This is the analytic solution for a layered earth under
    plane-wave excitation.

    Z_n = Z_{n,intrinsic}  (bottom layer)
    Z_j = Z_{j,intr} * (Z_{j+1} + Z_{j,intr} * tanh(ik_j * h_j)) /
                        (Z_{j,intr} + Z_{j+1} * tanh(ik_j * h_j))

    where k_j = sqrt(iωμσ_j) and Z_{j,intr} = iωμ/k_j.

    Parameters
    ----------
    frequencies : np.ndarray  Frequencies [Hz].
    thicknesses : list        Layer thicknesses [m] (N-1 for N layers).
    resistivities : list      Layer resistivities [Ω·m] (N layers).

    Returns
    -------
    Z : np.ndarray           Complex impedance at each frequency.
    app_res : np.ndarray     Apparent resistivity [Ω·m].
    phase : np.ndarray       Phase [degrees].
    """
    mu0 = 4 * np.pi * 1e-7  # H/m
    n_layers = len(resistivities)
    
    frequencies = np.atleast_1d(frequencies)
    Z = np.zeros(len(frequencies), dtype=complex)

    for fi, freq in enumerate(frequencies):
        omega = 2 * np.pi * freq

        # Intrinsic impedance and propagation constant for each layer
        sigma = [1.0 / r for r in resistivities]
        k = [np.sqrt(1j * omega * mu0 * s) for s in sigma]
        Z_intr = [1j * omega * mu0 / kk for kk in k]

        # Start from bottom (half-space)
        Z_below = Z_intr[-1]

        # Recurse upward
        for j in range(n_layers - 2, -1, -1):
            h = thicknesses[j]
            arg = k[j] * h
            # Numerical stability: clip argument
            if np.abs(arg) > 500:
                tanh_val = 1.0 if arg.real > 0 else -1.0
            else:
                tanh_val = np.tanh(arg)

            Z_below = Z_intr[j] * (Z_below + Z_intr[j] * tanh_val) / \
                      (Z_intr[j] + Z_below * tanh_val)

        Z[fi] = Z_below

    app_res = np.abs(Z) ** 2 / (2 * np.pi * frequencies * mu0)
    phase = np.degrees(np.angle(Z))

    return Z, app_res, phase
