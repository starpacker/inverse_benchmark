import matplotlib

matplotlib.use('Agg')

import os

import numpy as np

from skimage.transform import radon, iradon, resize

from skimage.restoration import denoise_tv_chambolle

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

def forward_operator(x, theta):
    """
    Radon transform (forward projection) using skimage.
    
    Args:
        x: Input image (N x N)
        theta: Projection angles
        
    Returns:
        y_pred: Sinogram (forward projection of x)
    """
    return radon(x, theta=theta, circle=True)

def run_inversion(sinogram, theta, output_size, adjoint_scale, method='TV-FISTA',
                  cgls_iter=30, fista_iter=100, lam_tv=0.003):
    """
    Run CT reconstruction using the specified method.
    
    Args:
        sinogram: Noisy sinogram data
        theta: Projection angles
        output_size: Size of output reconstruction
        adjoint_scale: Calibrated adjoint scaling factor
        method: 'FBP', 'CGLS', or 'TV-FISTA'
        cgls_iter: Number of CGLS iterations
        fista_iter: Number of FISTA iterations
        lam_tv: TV regularization parameter
        
    Returns:
        reconstruction: Reconstructed image
    """
    
    def back_project_scaled(sino):
        """Scaled adjoint operator (back projection)."""
        return iradon(sino, theta=theta, output_size=output_size,
                      filter_name=None, circle=True) * adjoint_scale
    
    if method == 'FBP':
        # Filtered Back-Projection (standard analytical method)
        print("\n[FBP] Running Filtered Back-Projection...")
        reconstruction = iradon(sinogram, theta=theta, output_size=output_size,
                                filter_name='ramp', circle=True)
        return reconstruction
    
    elif method == 'CGLS':
        # Conjugate Gradient Least Squares
        # Solves min_x || A x - b ||^2 where A is the Radon transform
        print(f"\n[CGLS] Running CGLS reconstruction ({cgls_iter} iterations)...")
        
        b = sinogram.copy()
        x = np.zeros((output_size, output_size), dtype=np.float64)
        
        r = b - forward_operator(x, theta)
        s = back_project_scaled(r)
        p = s.copy()
        gamma = np.sum(s ** 2)
        gamma0 = gamma
        
        for k in range(cgls_iter):
            Ap = forward_operator(p, theta)
            alpha = gamma / (np.sum(Ap ** 2) + 1e-30)
            
            x += alpha * p
            r -= alpha * Ap
            
            s = back_project_scaled(r)
            gamma_new = np.sum(s ** 2)
            
            beta = gamma_new / (gamma + 1e-30)
            p = s + beta * p
            gamma = gamma_new
            
            if k % 10 == 0 or k == cgls_iter - 1:
                res_norm = np.sqrt(np.sum(r ** 2))
                print(f"    iter {k:3d}: ||r||={res_norm:.4e}")
            
            if gamma < 1e-12 * gamma0:
                print(f"  CGLS converged at iteration {k+1}")
                break
        
        return x
    
    elif method == 'TV-FISTA':
        # TV-regularised reconstruction using FISTA + TV denoising
        # Solves: min_x (1/2) || A x - b ||^2 + λ_TV * TV(x)
        print(f"\n[TV-FISTA] Running TV-FISTA reconstruction ({fista_iter} iterations, λ_TV={lam_tv})...")
        
        # Estimate Lipschitz constant via power iteration
        rng = np.random.RandomState(42)
        v = rng.randn(output_size, output_size)
        for _ in range(30):
            Av = forward_operator(v, theta)
            ATAv = back_project_scaled(Av)
            norm_v = np.sqrt(np.sum(ATAv ** 2))
            v = ATAv / (norm_v + 1e-30)
        L = norm_v
        step_size = 0.9 / L
        print(f"  Lipschitz L ≈ {L:.2f}, step = {step_size:.6f}")
        
        # FISTA with warm start from FBP
        b = sinogram.copy()
        x = iradon(sinogram, theta=theta, output_size=output_size,
                   filter_name='ramp', circle=True)
        y = x.copy()
        t = 1.0
        
        for k in range(fista_iter):
            residual = forward_operator(y, theta) - b
            grad = back_project_scaled(residual)
            
            # Gradient step
            x_new = y - step_size * grad
            
            # TV proximal step
            x_new = denoise_tv_chambolle(x_new, weight=lam_tv * step_size,
                                          max_num_iter=50)
            
            # FISTA momentum
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            y = x_new + (t - 1) / t_new * (x_new - x)
            
            x = x_new
            t = t_new
            
            if k % 15 == 0 or k == fista_iter - 1:
                res_norm = np.sqrt(np.sum(residual ** 2))
                print(f"    iter {k:3d}: ||r||={res_norm:.4e}")
        
        return x
    
    else:
        raise ValueError(f"Unknown method: {method}")
