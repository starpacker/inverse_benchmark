import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

from scipy.sparse import lil_matrix

from scipy.sparse.linalg import spsolve

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_and_preprocess_data(k_train_values, k_test_values, nx, ny):
    """
    Generate training and test snapshots for parametric 2D heat conduction.
    
    Parameters
    ----------
    k_train_values : array-like
        Training thermal conductivity values.
    k_test_values : array-like
        Test thermal conductivity values.
    nx, ny : int
        Grid resolution.
    
    Returns
    -------
    params_train : ndarray of shape (n_train, 1)
    snapshots_train : ndarray of shape (n_train, nx*ny)
    params_test : ndarray of shape (n_test, 1)
    snapshots_test : ndarray of shape (n_test, nx*ny)
    grid_info : dict with grid parameters
    """
    dx = 1.0 / (nx + 1)
    dy = 1.0 / (ny + 1)
    
    x = np.linspace(dx, 1.0 - dx, nx)
    y = np.linspace(dy, 1.0 - dy, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    grid_info = {
        'nx': nx,
        'ny': ny,
        'dx': dx,
        'dy': dy,
        'X': X,
        'Y': Y
    }
    
    snapshots_train = []
    for k in k_train_values:
        T = forward_operator(k, nx, ny, grid_info)
        snapshots_train.append(T)
    
    snapshots_test = []
    for k in k_test_values:
        T = forward_operator(k, nx, ny, grid_info)
        snapshots_test.append(T)
    
    params_train = np.array(k_train_values).reshape(-1, 1)
    snapshots_train = np.array(snapshots_train)
    params_test = np.array(k_test_values).reshape(-1, 1)
    snapshots_test = np.array(snapshots_test)
    
    return params_train, snapshots_train, params_test, snapshots_test, grid_info

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
