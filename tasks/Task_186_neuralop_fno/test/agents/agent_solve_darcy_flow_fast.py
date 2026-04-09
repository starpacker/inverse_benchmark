import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

import sys

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")

if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(WORKING_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def solve_darcy_flow_fast(a, f_source=1.0):
    """
    Fast Darcy flow solver using scipy sparse solver.
    -∇·(a(x)∇u(x)) = f on [0,1]^2, u=0 on boundary.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    
    n = a.shape[0]
    h = 1.0 / (n + 1)
    N = n * n
    
    rows, cols, vals = [], [], []
    rhs = np.zeros(N)
    
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            rhs[idx] = f_source * h**2
            
            center_val = 0.0
            
            if j < n - 1:
                a_e = 0.5 * (a[i, j] + a[i, j+1])
                rows.append(idx); cols.append(idx + 1); vals.append(a_e)
                center_val += a_e
            else:
                center_val += a[i, j]
                
            if j > 0:
                a_w = 0.5 * (a[i, j] + a[i, j-1])
                rows.append(idx); cols.append(idx - 1); vals.append(a_w)
                center_val += a_w
            else:
                center_val += a[i, j]
                
            if i < n - 1:
                a_s = 0.5 * (a[i, j] + a[i+1, j])
                rows.append(idx); cols.append(idx + n); vals.append(a_s)
                center_val += a_s
            else:
                center_val += a[i, j]
                
            if i > 0:
                a_n = 0.5 * (a[i, j] + a[i-1, j])
                rows.append(idx); cols.append(idx - n); vals.append(a_n)
                center_val += a_n
            else:
                center_val += a[i, j]
            
            rows.append(idx); cols.append(idx); vals.append(-center_val)
    
    A_mat = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
    u_vec = spsolve(A_mat, -rhs)
    u = u_vec.reshape(n, n).astype(np.float32)
    
    return u
