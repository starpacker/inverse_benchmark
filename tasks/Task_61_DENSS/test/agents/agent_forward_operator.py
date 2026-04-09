import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.fft import fftn, ifftn, fftshift, ifftshift, fftfreq

def forward_operator(density, voxel_size, n_q, q_max):
    """
    Compute 1D SAXS profile I(q) from 3D electron density.

    I(q) = spherical_average( |FFT{ρ(r)}|² )

    Parameters
    ----------
    density : ndarray
        3D electron density array.
    voxel_size : float
        Voxel size in Angstroms.
    n_q : int
        Number of q bins.
    q_max : float
        Maximum q value in inverse Angstroms.

    Returns
    -------
    q_bins : ndarray
        q values in inverse Angstroms.
    I_q : ndarray
        Scattering intensity (normalized).
    """
    N = density.shape[0]

    # 3D FFT
    F = fftshift(fftn(ifftshift(density)))
    I_3d = np.abs(F) ** 2

    # q-grid
    freq = fftfreq(N, d=voxel_size)
    freq = fftshift(freq)
    qx, qy, qz = np.meshgrid(freq, freq, freq, indexing='ij')
    q_3d = 2 * np.pi * np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)

    # Radial average (spherical shells)
    q_bins = np.linspace(0.01, q_max, n_q)
    dq = q_bins[1] - q_bins[0]
    I_q = np.zeros(n_q)

    for i, qc in enumerate(q_bins):
        mask = (q_3d >= qc - dq / 2) & (q_3d < qc + dq / 2)
        if mask.sum() > 0:
            I_q[i] = np.mean(I_3d[mask])

    # Normalise
    if I_q.max() > 0:
        I_q = I_q / I_q.max()

    return q_bins, I_q
