import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

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
