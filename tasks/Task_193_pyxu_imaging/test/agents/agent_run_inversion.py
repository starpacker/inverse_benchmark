import numpy as np

import matplotlib

matplotlib.use('Agg')

import os

import sys

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")

if os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

import pyxu.operator as pxo

import pyxu.opt.solver as pxs

import pyxu.opt.stop as pxst

def run_inversion(y_observed, img_shape, kernel, lambda_tv, max_iter):
    """
    Solve the TV-regularized deconvolution problem using Condat-Vu primal-dual splitting.
    
    Solves: min_x 0.5*||H*x - y||_2^2 + lambda * ||nabla(x)||_1
    
    Parameters:
    -----------
    y_observed : np.ndarray
        Observed (blurred + noisy) data (1D flattened)
    img_shape : tuple
        Shape of the 2D image (height, width)
    kernel : np.ndarray
        2D Gaussian blur kernel
    lambda_tv : float
        Total Variation regularization weight
    max_iter : int
        Maximum number of solver iterations
        
    Returns:
    --------
    x_reconstructed : np.ndarray
        Reconstructed image (2D)
    """
    N = img_shape[0] * img_shape[1]
    
    # Build forward operator H
    H = pxo.Convolve(
        arg_shape=img_shape,
        kernel=kernel,
        center=(kernel.shape[0]//2, kernel.shape[1]//2),
        mode="constant",
    )
    
    # Data fidelity: f(x) = 0.5 * ||Hx - y||^2
    sl2 = 0.5 * pxo.SquaredL2Norm(dim=N)
    f = sl2.asloss(y_observed) * H
    
    # TV regularizer: h(z) = lambda * ||z||_1, K = Gradient
    grad_op = pxo.Gradient(arg_shape=img_shape)
    h = lambda_tv * pxo.L1Norm(dim=grad_op.codim)
    K = grad_op
    
    print(f"  Forward op H: dim={H.dim}, codim={H.codim}")
    print(f"  Gradient K: dim={K.dim}, codim={K.codim}")
    print(f"  f: 0.5*||Hx-y||^2  (differentiable)")
    print(f"  h: {lambda_tv} * ||z||_1  (proximable)")
    print(f"  K: Gradient (finite differences)")
    
    # Condat-Vu solver
    solver = pxs.CV(f=f, h=h, K=K, show_progress=False)
    stop = pxst.MaxIter(n=max_iter)
    
    # Initial guess: degraded observation
    x0 = y_observed.copy()
    
    print(f"  Running Condat-Vu solver (max_iter={max_iter})...")
    solver.fit(x0=x0, stop_crit=stop)
    
    x_sol = solver.solution()
    print(f"  Solver finished. Solution shape: {x_sol.shape}")
    
    # Clip to valid range and reshape
    x_reconstructed = np.clip(x_sol, 0, 1).reshape(img_shape)
    print(f"  Reconstruction range: [{x_reconstructed.min():.4f}, {x_reconstructed.max():.4f}]")
    
    return x_reconstructed
