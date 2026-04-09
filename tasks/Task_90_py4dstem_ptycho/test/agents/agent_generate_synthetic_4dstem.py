import numpy as np

import matplotlib

matplotlib.use("Agg")

def generate_synthetic_4dstem(gt_phase, probe_array, positions_px, dose):
    """
    Forward-simulate 4D-STEM data using the exact same coordinate
    convention that py4DSTEM's SingleslicePtychography uses internally.
    """
    from py4DSTEM.process.phase.utils import fft_shift

    Sx, Sy = probe_array.shape
    Px, Py = gt_phase.shape
    N = positions_px.shape[0]

    gt_object = np.exp(1j * gt_phase).astype(np.complex64)

    x_ind = np.fft.fftfreq(Sx, d=1.0 / Sx).astype(int)
    y_ind = np.fft.fftfreq(Sy, d=1.0 / Sy).astype(int)

    data = np.zeros((N, Sx, Sy), dtype=np.float64)

    for i in range(N):
        pos = positions_px[i]
        r0 = int(np.round(pos[0]))
        c0 = int(np.round(pos[1]))
        pos_frac = pos - np.round(pos)

        row_idx = (r0 + x_ind) % Px
        col_idx = (c0 + y_ind) % Py
        obj_patch = gt_object[np.ix_(row_idx, col_idx)]

        shifted_probe = fft_shift(
            probe_array, pos_frac.reshape(1, 2), np
        )[0]

        overlap = shifted_probe * obj_patch
        fourier_overlap = np.fft.fft2(overlap)
        dp_corner = np.abs(fourier_overlap) ** 2

        dp_centered = np.fft.fftshift(dp_corner)

        dp_scaled = dp_centered / (dp_centered.sum() + 1e-30) * dose
        data[i] = np.random.poisson(
            np.clip(dp_scaled, 0, None)
        ).astype(np.float64)

    return data.astype(np.float32)
