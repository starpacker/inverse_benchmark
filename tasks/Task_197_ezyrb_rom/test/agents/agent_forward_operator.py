import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.sparse import lil_matrix

from scipy.sparse.linalg import spsolve

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def forward_operator(k, nx, ny, grid_info=None):
    """
    Forward operator: Solve steady-state 2D heat equation.
    Maps thermal conductivity parameter k to temperature field.
    
    -k * (d^2T/dx^2 + d^2T/dy^2) = f(x,y)
    on [0,1]x[0,1] with Dirichlet BCs using finite differences.
    
    Parameters
    ----------
    k : float
        Thermal conductivity parameter.
    nx, ny : int
        Grid resolution.
    grid_info : dict or None
        Precomputed grid information (optional).
    
    Returns
    -------
    T : ndarray of shape (nx*ny,)
        Flattened temperature field.
    """
    dx = 1.0 / (nx + 1)
    dy = 1.0 / (ny + 1)
    
    x = np.linspace(dx, 1.0 - dx, nx)
    y = np.linspace(dy, 1.0 - dy, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    f = 100.0 * np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / (2 * 0.05**2))
    f += 50.0 * k * np.sin(2 * np.pi * X) * np.sin(np.pi * Y)
    
    N = nx * ny
    h = dx
    coeff_diag = 4.0 * k / h**2
    coeff_off = -k / h**2
    
    A = lil_matrix((N, N))
    rhs = f.flatten()
    
    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            A[idx, idx] = coeff_diag
            
            if i > 0:
                A[idx, (i - 1) * ny + j] = coeff_off
            if i < nx - 1:
                A[idx, (i + 1) * ny + j] = coeff_off
            if j > 0:
                A[idx, i * ny + (j - 1)] = coeff_off
            if j < ny - 1:
                A[idx, i * ny + (j + 1)] = coeff_off
    
    A = A.tocsr()
    T = spsolve(A, rhs)
    return T
