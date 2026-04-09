import warnings

import matplotlib

matplotlib.use('Agg')

import odl

warnings.filterwarnings('ignore')

def run_inversion(data, tv_lambda=0.0002, niter_pdhg=600, niter_cgls=30):
    """
    Run CT reconstruction using multiple methods.
    
    Implements three reconstruction algorithms:
    1. FBP (Filtered Back Projection) - analytic baseline
    2. CGLS (Conjugate Gradient Least Squares) - iterative
    3. TV-PDHG (Total Variation via Primal-Dual Hybrid Gradient) - regularized
    
    Parameters
    ----------
    data : dict
        Data dictionary from load_and_preprocess_data containing:
        - 'reco_space': reconstruction space
        - 'ray_transform': forward operator
        - 'sinogram': noisy measurement data
    tv_lambda : float
        Regularization parameter for TV penalty.
    niter_pdhg : int
        Number of PDHG iterations.
    niter_cgls : int
        Number of CGLS iterations.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'fbp': FBP reconstruction (numpy array)
        - 'cgls': CGLS reconstruction (numpy array)
        - 'pdhg': TV-PDHG reconstruction (numpy array)
        - 'primary': Primary result (TV-PDHG) as numpy array
        - 'parameters': dict of algorithm parameters
    """
    reco_space = data['reco_space']
    ray_transform = data['ray_transform']
    sinogram = data['sinogram']
    
    # ─── FBP reconstruction ─────────────────────────────────────────────────
    print('[FBP] Running Filtered Back Projection …')
    fbp_op = odl.tomo.fbp_op(ray_transform)
    x_fbp = fbp_op(sinogram)
    recon_fbp = x_fbp.asarray()
    
    # ─── CGLS reconstruction ────────────────────────────────────────────────
    print('[CGLS] Running Conjugate Gradient (normal equations) …')
    x_cgls = ray_transform.domain.zero()
    odl.solvers.conjugate_gradient_normal(
        ray_transform, x_cgls, sinogram, niter=niter_cgls,
        callback=odl.solvers.CallbackPrintIteration(step=10, fmt='  CGLS iter {}')
    )
    recon_cgls = x_cgls.asarray()
    
    # ─── TV-PDHG reconstruction ─────────────────────────────────────────────
    print('[TV-PDHG] Running TV-regularised PDHG (warm-started from FBP) …')
    gradient = odl.Gradient(reco_space)
    
    # Broadcast operator: K = [ray_transform; gradient]
    op = odl.BroadcastOperator(ray_transform, gradient)
    
    # Primal functional: f = 0 (no explicit constraint on x)
    f = odl.solvers.ZeroFunctional(op.domain)
    
    # Dual functionals
    # Data fidelity: 0.5 * ||Ax - b||_2^2
    l2_term = odl.solvers.L2NormSquared(ray_transform.range).translated(sinogram)
    # TV penalty: lambda * ||grad(x)||_1 (isotropic)
    l1_term = tv_lambda * odl.solvers.L1Norm(gradient.range)
    
    g = odl.solvers.SeparableSum(l2_term, l1_term)
    
    # Step sizes (tau * sigma * ||K||^2 < 1 required for convergence)
    op_norm = 1.1 * odl.power_method_opnorm(op, maxiter=20)
    tau = 1.0 / op_norm
    sigma = 1.0 / op_norm
    
    # Warm-start from FBP for faster convergence
    x_pdhg = x_fbp.copy()
    
    odl.solvers.pdhg(
        x_pdhg, f, g, op,
        niter=niter_pdhg, tau=tau, sigma=sigma,
        callback=odl.solvers.CallbackPrintIteration(step=100, fmt='  PDHG iter {}')
    )
    recon_pdhg = x_pdhg.asarray()
    
    return {
        'fbp': recon_fbp,
        'cgls': recon_cgls,
        'pdhg': recon_pdhg,
        'primary': recon_pdhg,
        'parameters': {
            'tv_lambda': tv_lambda,
            'niter_pdhg': niter_pdhg,
            'niter_cgls': niter_cgls
        }
    }
