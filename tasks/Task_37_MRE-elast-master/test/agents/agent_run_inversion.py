import time

import numpy as np

from pyunlocbox import functions, solvers

def _nodeneighbor(triangles, N):
    T = [[] for _ in range(N)]
    for elem in triangles:
        for node in elem:
            T[node].extend(elem)
    for i in range(N):
        T[i] = list(set(T[i]))
    T1 = np.zeros((N, 10))
    for i in range(N):
        length = min(len(T[i]), 10)
        T1[i, :length] = T[i][:length]
    return T1

def _ggradient(E, A):
    N = np.size(E)
    grad1 = np.zeros((N, 1))
    for n in range(N):
        S1 = 0
        for k in range(10):
            if A[n, k] != 0:
                neighbor_idx = int(A[n, k])
                if neighbor_idx < N:
                    S1 = S1 + (E[n] - E[neighbor_idx])**2
        grad1[n] = np.sqrt(S1)
    if np.max(grad1) != 0:
        result1 = grad1 / np.max(np.ceil(grad1))
    else:
        result1 = grad1
    return np.ravel(result1)

def run_inversion(disp_measured, E_initial_guess, matTens, Tens, KT, triangles, vertices, fxy, params):
    """
    Performs the Proximal Optimization to reconstruct Stiffness (E) from Displacement (disp_measured).
    """
    start = time.time()
    
    # Unpack params
    tau1 = params.get('tau1', 0.94)
    tau2 = params.get('tau2', 0.06)
    tau3 = params.get('tau3', 0.01)
    step = params.get('step', 0.16 * 0.6)
    maxit = params.get('maxit', 20)
    outer_itr = params.get('outer_itr', 1)
    
    um = disp_measured
    N = matTens.shape[0] # Number of nodes (E is defined on nodes in this formulation)
    
    # Construct Operator D for ||y - D x||
    # Dm = matTens @ um -> (N, 2N)
    Dm = matTens @ um
    D2 = (Dm.T) / 3 
    
    # Target vector
    ft = fxy - KT @ um * 1e-5
    
    # Initial Solution
    sol = E_initial_guess
    
    # TV Operator Helper
    T1 = _nodeneighbor(triangles, N)
    g = lambda x: _ggradient(x, T1)
    
    # Covariance for Mahalanobis distance term
    # We need a GK0 estimate to build Gamma. 
    # The original code updates GK0 inside the outer loop.
    
    yy2 = np.ravel(ft)
    N_cov = np.zeros((2 * N, 2 * N))
    for j in np.arange(0, 2 * N, 2):
        N_cov[j, j] = 3
        N_cov[j + 1, j + 1] = 1
        
    w = np.ones((2 * N, 1))
    w[:102 * 2, :] = 0.99
    
    for j in range(outer_itr):
        GK0 = (np.squeeze(Tens @ sol / 3) + 1e-5 * KT)
        
        # Build Gamma
        gamma = GK0 @ N_cov @ GK0.T
        gamma += np.eye(gamma.shape[1]) * 1e-5
        gamma_inv = np.linalg.inv(gamma)
        
        # Define Functions for PyUnlocBox
        
        # 1. Quadratic Data Fidelity with Covariance
        f8 = functions.func(lambda_=tau1)
        f8._eval = lambda x: 0.5 * (yy2 - D2 @ x).T @ gamma_inv @ (yy2 - D2 @ x) * 1e-4
        f8._grad = lambda x: -1 * D2.T @ gamma_inv @ (yy2 - D2 @ x) * 1e-4
        
        # 2. TV / Smoothness
        f3 = functions.norm_l1(A=g, At=None, dim=1, y=np.zeros(N), lambda_=tau2)
        
        # 3. Box Constraint [0, 0.7] (Normalized Stiffness)
        f2 = functions.func()
        f2._eval = lambda x: 0
        f2._prox = lambda x, T: np.clip(x, 0, 0.7)
        
        # Solver
        solver = solvers.generalized_forward_backward(step=step)
        
        # Solve
        print(f"  Starting Optimization Loop {j+1}/{outer_itr}...")
        ret = solvers.solve([f8, f3, f2], np.copy(sol), solver, rtol=1e-15, maxit=maxit, verbosity='LOW')
        sol = ret['sol']
        
        if np.any(np.isnan(sol)) or np.mean(sol) > 1:
            print("  Warning: Solver diverged or produced invalid values.")
            break
            
    print("Inversion time: %.4f seconds" % (time.time() - start))
    return sol
