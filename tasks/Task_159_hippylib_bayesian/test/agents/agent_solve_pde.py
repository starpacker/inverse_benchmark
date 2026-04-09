from scipy import sparse

from scipy.sparse.linalg import spsolve

import matplotlib

matplotlib.use('Agg')

def build_diffusion_matrix(kappa_2d, h):
    """
    Build sparse stiffness matrix for -div(kappa * grad(u)) = f
    on interior nodes with spacing h.
    Uses harmonic averaging of kappa at cell interfaces.
    """
    Ny, Nx = kappa_2d.shape
    N = Ny * Nx
    rows, cols, vals = [], [], []

    for iy in range(Ny):
        for ix in range(Nx):
            i = iy * Nx + ix
            diag_val = 0.0

            for diy, dix in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                niy, nix = iy + diy, ix + dix
                if 0 <= niy < Ny and 0 <= nix < Nx:
                    k_avg = 2.0 * kappa_2d[iy, ix] * kappa_2d[niy, nix] / (
                        kappa_2d[iy, ix] + kappa_2d[niy, nix] + 1e-30)
                    coeff = k_avg / (h * h)
                    j = niy * Nx + nix
                    rows.append(i)
                    cols.append(j)
                    vals.append(-coeff)
                    diag_val += coeff
                else:
                    diag_val += kappa_2d[iy, ix] / (h * h)

            rows.append(i)
            cols.append(i)
            vals.append(diag_val)

    return sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))

def solve_pde(kappa_2d, source_2d, h):
    """Solve -div(kappa grad u) = f for u on interior nodes."""
    A = build_diffusion_matrix(kappa_2d, h)
    return spsolve(A, source_2d.ravel())
