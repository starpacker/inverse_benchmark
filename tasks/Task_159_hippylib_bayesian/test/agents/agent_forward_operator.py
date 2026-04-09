import numpy as np

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

def kappa_from_params(params, X, Y, centers, sigma_basis, kappa_bg=1.0):
    """
    kappa(x,y) = kappa_bg + sum_i params[i] * G_i(x,y)
    where G_i is a Gaussian centered at centers[i].
    """
    kappa = np.full_like(X, kappa_bg)
    for i in range(len(params)):
        cx, cy = centers[i]
        kappa += params[i] * np.exp(
            -((X - cx)**2 + (Y - cy)**2) / (2.0 * sigma_basis**2)
        )
    return np.maximum(kappa, 0.1)

def forward_operator(params, data):
    """
    Forward operator: maps parameters to predicted observations.
    
    Given kappa parameterization -> solve PDE -> apply observation operator
    
    Args:
        params: numpy array of basis function coefficients
        data: dict containing grid info, sources, observation operator, etc.
        
    Returns:
        list of predicted observations for each source experiment
    """
    X = data['X']
    Y = data['Y']
    centers = data['centers']
    sigma_basis = data['sigma_basis']
    kappa_bg = data['kappa_bg']
    sources = data['sources']
    h = data['h']
    B = data['B']
    
    kappa_2d = kappa_from_params(params, X, Y, centers, sigma_basis, kappa_bg)
    
    predictions = []
    for src in sources:
        u = solve_pde(kappa_2d, src, h)
        y_pred = B @ u
        predictions.append(y_pred)
    
    return predictions
