import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.optimize import minimize

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

NOISE_FLOOR = 0.01

NOISE_PCT = 0.02

N_LAYERS = 30

MIN_DEPTH = 10.0

MAX_DEPTH = 5000.0

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

def run_inversion(data_dict):
    """
    1D MT inversion using Occam's razor approach with analytical Jacobian.
    Tikhonov regularisation with smoothness constraint in log-resistivity space.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary from load_and_preprocess_data containing frequencies and Z_noisy.
    
    Returns
    -------
    result_dict : dict
        Dictionary containing:
        - rec_resistivities: recovered resistivity profile
        - depths: depth array
        - layer_thicknesses: thickness of each layer
        - Z_pred: predicted impedance
        - rho_pred: predicted apparent resistivity
        - phi_pred: predicted phase
    """
    frequencies = data_dict['frequencies']
    Z_noisy = data_dict['Z_noisy']
    
    mu0 = 4 * np.pi * 1e-7
    n_freq = len(frequencies)
    n_layers = N_LAYERS
    ln10 = np.log(10.0)
    
    # Create logarithmically spaced depth layers
    layer_thicknesses = np.logspace(
        np.log10(MIN_DEPTH), np.log10(MAX_DEPTH / N_LAYERS), N_LAYERS - 1
    )
    depths = np.cumsum(layer_thicknesses)

    # Inversion in log10-resistivity space
    m0 = np.log10(np.ones(N_LAYERS) * 100.0)  # starting model: 100 Ω·m

    # Smoothness matrix (first-order finite difference)
    D = np.zeros((N_LAYERS - 1, N_LAYERS))
    for i in range(N_LAYERS - 1):
        D[i, i] = -1
        D[i, i + 1] = 1

    # Precompute DtD for regularisation gradient
    DtD = D.T @ D

    # Noise estimate
    Z_std = NOISE_FLOOR * np.abs(Z_noisy) + NOISE_PCT * np.abs(Z_noisy)

    def forward_with_analytical_grad(m):
        """
        Compute impedance Z and analytical dZ/dm via differentiation of
        Wait's recursion w.r.t. log10(rho).
        """
        resistivities = 10.0 ** m

        Z_pred = np.zeros(n_freq, dtype=complex)
        J = np.zeros((n_freq, n_layers), dtype=complex)

        for fi in range(n_freq):
            omega = 2 * np.pi * frequencies[fi]

            sigma = 1.0 / resistivities
            k = np.sqrt(1j * omega * mu0 * sigma)
            Z_intr = 1j * omega * mu0 / k

            # Forward pass: store Z at each interface
            Z_at = np.zeros(n_layers, dtype=complex)
            tanh_v = np.zeros(n_layers - 1, dtype=complex)
            Z_at[-1] = Z_intr[-1]

            for j in range(n_layers - 2, -1, -1):
                h = layer_thicknesses[j]
                arg = k[j] * h
                if np.abs(arg) > 500:
                    tanh_v[j] = 1.0 if arg.real > 0 else -1.0
                else:
                    tanh_v[j] = np.tanh(arg)
                Zb = Z_at[j + 1]
                tv = tanh_v[j]
                D_den = Z_intr[j] + Zb * tv
                Z_at[j] = Z_intr[j] * (Zb + Z_intr[j] * tv) / D_den

            Z_pred[fi] = Z_at[0]

            # Compute dZ_j/dZ_{j+1} for chain rule
            dZ_dZnext = np.zeros(n_layers - 1, dtype=complex)
            for j in range(n_layers - 2, -1, -1):
                tv = tanh_v[j]
                Zb = Z_at[j + 1]
                Zi = Z_intr[j]
                D_den = Zi + Zb * tv
                N_num = Zb + Zi * tv
                dZ_dZnext[j] = Zi * (D_den - N_num * tv) / (D_den ** 2)

            # Compute chain product
            chain_prod = np.ones(n_layers, dtype=complex)
            for p in range(1, n_layers):
                chain_prod[p] = chain_prod[p - 1] * dZ_dZnext[p - 1]

            # Compute local dZ_p/dm_p for each layer p
            for p in range(n_layers):
                dZintr_dm = Z_intr[p] * ln10 / 2.0
                dk_dm = -k[p] * ln10 / 2.0

                if p == n_layers - 1:
                    dZp_local = dZintr_dm
                else:
                    tv = tanh_v[p]
                    Zb = Z_at[p + 1]
                    Zi = Z_intr[p]
                    h = layer_thicknesses[p]
                    N_num = Zb + Zi * tv
                    D_den = Zi + Zb * tv

                    dt_dm = (1.0 - tv ** 2) * h * dk_dm
                    dN = dZintr_dm * tv + Zi * dt_dm
                    dD = dZintr_dm + Zb * dt_dm

                    dZp_local = dZintr_dm * N_num / D_den + \
                                Zi * (dN * D_den - N_num * dD) / (D_den ** 2)

                J[fi, p] = chain_prod[p] * dZp_local

        return Z_pred, J

    def objective_and_grad(m, beta=1.0):
        Z_pred, J = forward_with_analytical_grad(m)

        # Residuals
        res_real = (Z_noisy.real - Z_pred.real) / Z_std
        res_imag = (Z_noisy.imag - Z_pred.imag) / Z_std

        misfit = float(np.sum(res_real ** 2 + res_imag ** 2))
        reg = float(np.sum((D @ m) ** 2))
        obj = misfit + beta * reg

        # Gradient via matrix ops
        grad_misfit = -2.0 * (
            (res_real / Z_std) @ J.real + (res_imag / Z_std) @ J.imag
        )
        grad_reg = 2.0 * DtD @ m
        grad = grad_misfit + beta * grad_reg

        return obj, grad

    # Multi-stage inversion with decreasing regularisation
    print("[RECON] Occam 1D MT inversion (log-resistivity, analytical grad) ...")
    m_current = m0.copy()
    betas = [100, 30, 10, 3, 1, 0.3, 0.1, 0.03, 0.01]

    for beta in betas:
        result = minimize(
            objective_and_grad, m_current, args=(beta,),
            method='L-BFGS-B', jac=True,
            bounds=[(-1, 5)] * N_LAYERS,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        m_current = result.x
        chi2 = result.fun
        print(f"[RECON]   β={beta:8.2f}  χ²={chi2:.2f}")

    rec_res = 10 ** m_current

    # Predicted data
    Z_pred, rho_pred, phi_pred = forward_operator(
        frequencies, layer_thicknesses.tolist(), rec_res.tolist()
    )

    result_dict = {
        'rec_resistivities': rec_res,
        'depths': depths,
        'layer_thicknesses': layer_thicknesses,
        'Z_pred': Z_pred,
        'rho_pred': rho_pred,
        'phi_pred': phi_pred,
    }
    
    return result_dict
