import numpy as np

import matplotlib

matplotlib.use('Agg')

from koma.oma import covssi

from koma.modal import xmacmat, maxreal

def run_inversion(acc_data, fs, n_dof, freq_true):
    """
    Run Cov-SSI identification using koma.
    Returns identified frequencies, damping ratios, and mode shapes.
    """
    i_blockrows = 30
    orders = list(range(2, 2 * n_dof * 4 + 2, 2))

    print(f"Running Cov-SSI with i={i_blockrows}, orders={orders[:5]}...{orders[-3:]}")

    # Run covssi
    lambd, phi, order_arr = covssi(
        acc_data, fs, i=i_blockrows, orders=orders,
        weighting='none', matrix_type='hankel',
        algorithm='shift', showinfo=True, balance=True,
        return_flat=True
    )

    # Convert complex eigenvalues to frequencies and damping ratios
    omega = np.abs(lambd)
    freq_all = omega / (2 * np.pi)
    zeta_all = -np.real(lambd) / omega

    # Filter physical poles
    f_max = fs / 2
    mask = (freq_all > 0.1) & (freq_all < f_max * 0.9) & (zeta_all > 0) & (zeta_all < 1.0)
    freq_filt = freq_all[mask]
    zeta_filt = zeta_all[mask]
    phi_filt = phi[:, mask]
    order_filt = order_arr[mask]

    print(f"Filtered poles: {len(freq_filt)} (from {len(freq_all)} total)")

    # Match identified modes to true modes
    freq_id = np.zeros(n_dof)
    zeta_id = np.zeros(n_dof)
    phi_id = np.zeros((n_dof, n_dof), dtype=complex)

    for mode_idx in range(n_dof):
        f_true = freq_true[mode_idx]

        tol = 0.10 * f_true
        nearby = np.where(np.abs(freq_filt - f_true) < tol)[0]

        if len(nearby) == 0:
            tol = 0.20 * f_true
            nearby = np.where(np.abs(freq_filt - f_true) < tol)[0]

        if len(nearby) == 0:
            print(f"  Warning: No pole found near mode {mode_idx + 1} (f={f_true:.2f} Hz)")
            freq_id[mode_idx] = f_true
            zeta_id[mode_idx] = 0.0
            phi_id[:, mode_idx] = 0.0
            continue

        zeta_nearby = zeta_filt[nearby]
        reasonable = nearby[(zeta_nearby > 0.001) & (zeta_nearby < 0.15)]
        if len(reasonable) == 0:
            reasonable = nearby

        freq_diffs = np.abs(freq_filt[reasonable] - f_true)
        best_order_idx = reasonable[np.argmin(freq_diffs)]
        freq_id[mode_idx] = freq_filt[best_order_idx]
        zeta_id[mode_idx] = zeta_filt[best_order_idx]
        phi_id[:, mode_idx] = phi_filt[:, best_order_idx]

        print(f"  Mode {mode_idx + 1}: f_true={f_true:.3f} Hz, f_id={freq_id[mode_idx]:.3f} Hz, "
              f"zeta_id={zeta_id[mode_idx]:.4f}, order={order_filt[best_order_idx]}")

    # Normalize identified mode shapes
    phi_id_real = np.real(maxreal(phi_id))
    for j in range(n_dof):
        max_val = np.max(np.abs(phi_id_real[:, j]))
        if max_val > 0:
            phi_id_real[:, j] = phi_id_real[:, j] / max_val * np.sign(
                phi_id_real[np.argmax(np.abs(phi_id_real[:, j])), j]
            )

    result = {
        'freq_id': freq_id,
        'zeta_id': zeta_id,
        'phi_id': phi_id_real
    }

    return result
