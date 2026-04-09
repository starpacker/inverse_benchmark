import numpy as np


# --- Extracted Dependencies ---

def TV_denoiser(x, _lambda, n_iter_max):
    """
    Total Variation Denoiser (Chambolle's algorithm).
    """
    dt = 0.25
    N = x.shape
    idx = np.arange(1, N[0] + 1)
    idx[-1] = N[0] - 1
    iux = np.arange(-1, N[0] - 1)
    iux[0] = 0
    ir = np.arange(1, N[1] + 1)
    ir[-1] = N[1] - 1
    il = np.arange(-1, N[1] - 1)
    il[0] = 0
    p1 = np.zeros_like(x)
    p2 = np.zeros_like(x)
    divp = np.zeros_like(x)

    for i in range(n_iter_max):
        z = divp - x * _lambda
        z1 = z[:, ir, :] - z
        z2 = z[idx, :, :] - z
        denom_2d = 1 + dt * np.sqrt(np.sum(z1 ** 2 + z2 ** 2, 2))
        denom_3d = np.tile(denom_2d[:, :, np.newaxis], (1, 1, N[2]))
        p1 = (p1 + dt * z1) / denom_3d
        p2 = (p2 + dt * z2) / denom_3d
        divp = p1 - p1[:, il, :] + p2 - p2[iux, :, :]

    u = x - divp / _lambda
    return u
