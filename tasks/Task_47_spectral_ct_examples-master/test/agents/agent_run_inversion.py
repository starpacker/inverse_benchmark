import numpy as np

import scipy.linalg as spl

from scipy import signal

import odl

def estimate_cov(I1, I2):
    """
    Estimate the covariance of I1 and I2 using a Laplacian-like filter.
    This emphasizes high-frequency noise.
    """
    assert I1.shape == I2.shape
    H, W = I1.shape
    
    # Laplacian kernel to filter out smooth signal and keep noise
    M = np.array([[1, -2, 1],
                  [-2, 4., -2],
                  [1, -2, 1]])
    
    # Convolve and compute scalar product
    sigma = np.sum(signal.convolve2d(I1, M) * signal.convolve2d(I2, M))
    sigma /= (W * H - 1)
    
    # Normalization factor (empirical or derived from M)
    return sigma / 36.0

def cov_matrix(data):
    """
    Estimate the covariance matrix from data (stack of images/sinograms).
    data: (N, H, W)
    Returns: (N, N) covariance matrix.
    """
    n = len(data)
    cov_mat = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            cov_mat[i, j] = estimate_cov(data[i], data[j])
    return cov_mat

def run_inversion(data, space, geometry):
    """
    Sets up and solves the inverse problem using Douglas-Rachford Primal-Dual.
    
    Args:
        data: Noisy sinogram data (2, angles, detectors).
        space: ODL reconstruction space.
        geometry: ODL geometry.
        
    Returns:
        result: Reconstructed volume (2, H, W) numpy array.
    """
    # 1. Forward Operator Setup
    if odl.tomo.ASTRA_CUDA_AVAILABLE:
        impl = 'astra_cuda'
    else:
        impl = 'astra_cpu'
    ray_trafo = odl.tomo.RayTransform(space, geometry, impl=impl)
    A = odl.DiagonalOperator(ray_trafo, 2)
    
    # 2. Noise Estimation & Whitening
    cov_est = cov_matrix(data)
    w_mat = spl.fractional_matrix_power(cov_est, -0.5)
    
    # Whitening Operator
    I_range = odl.IdentityOperator(ray_trafo.range)
    W = odl.ProductSpaceOperator(np.multiply(w_mat, I_range))
    
    # Whitened Forward Operator and Data
    op = W * A
    rhs = W(data)
    
    # Data Matching Functional: || W(Ax - b) ||^2
    data_discrepancy = odl.solvers.L2NormSquared(A.range).translated(rhs)
    
    # 3. Regularization (Joint Prior: Nuclear Norm of Gradient)
    grad = odl.Gradient(space)
    L = odl.DiagonalOperator(grad, 2)
    lambda_reg = 0.15 
    regularizer = lambda_reg * odl.solvers.NuclearNorm(L.range)
    
    # 4. Solver Setup
    # Initial guess: FBP
    fbp_op = odl.tomo.fbp_op(ray_trafo, filter_type='Hann', frequency_scaling=0.7)
    x = A.domain.element([fbp_op(data[0]), fbp_op(data[1])])
    
    # Constraint: Non-negativity
    f_func = odl.solvers.IndicatorBox(A.domain, 0, np.inf)
    
    g_funcs = [data_discrepancy, regularizer]
    lin_ops = [op, L]
    
    # Step size estimation
    op_norm = odl.power_method_opnorm(op)
    grad_norm = odl.power_method_opnorm(grad)
    tau = 1.0
    sigma = (1.0/op_norm**2, 1.0/grad_norm**2)
    
    # Solve
    niter = 50
    callback = odl.solvers.CallbackPrintIteration()
    odl.solvers.douglas_rachford_pd(x, f_func, g_funcs, lin_ops, niter, 
                                    tau=tau, sigma=sigma, callback=callback)
    
    return x.asarray()
