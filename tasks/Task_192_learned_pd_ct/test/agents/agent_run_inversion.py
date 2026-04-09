import numpy as np

import matplotlib

matplotlib.use('Agg')

from skimage.transform import resize, radon, iradon

def run_inversion(sinogram_noisy, theta_angles, image_size=256, lam=0.02, niter_tv=300):
    """
    Run CT reconstruction via FBP + TV-PDHG.
    
    Stage 1: Filtered Back-Projection (FBP) for initial reconstruction
    Stage 2: Total Variation (TV) denoising via Chambolle-Pock PDHG
    
    Parameters:
    -----------
    sinogram_noisy : ndarray
        Noisy sinogram measurement
    theta_angles : ndarray
        Projection angles
    image_size : int
        Size of the reconstructed image
    lam : float
        TV regularization weight
    niter_tv : int
        Number of TV-PDHG iterations
    
    Returns:
    --------
    dict containing:
        - 'reconstruction': Final reconstructed image
        - 'fbp_reconstruction': Initial FBP reconstruction
    """
    
    # Helper functions for gradient and divergence
    def grad(x):
        """Discrete gradient with forward differences: returns (2, N, N)."""
        gx = np.zeros_like(x)
        gy = np.zeros_like(x)
        gx[:-1, :] = x[1:, :] - x[:-1, :]
        gy[:, :-1] = x[:, 1:] - x[:, :-1]
        return np.stack([gx, gy], axis=0)
    
    def div(p):
        """Discrete divergence = -∇^* (negative adjoint of gradient)."""
        px, py = p[0], p[1]
        dx = np.zeros_like(px)
        dy = np.zeros_like(py)
        dx[0, :] = px[0, :]
        dx[1:-1, :] = px[1:-1, :] - px[:-2, :]
        dx[-1, :] = -px[-2, :]
        dy[:, 0] = py[:, 0]
        dy[:, 1:-1] = py[:, 1:-1] - py[:, :-2]
        dy[:, -1] = -py[:, -1]
        return dx + dy
    
    # Stage 1: Filtered Back-Projection (FBP)
    fbp_recon = iradon(sinogram_noisy, theta=theta_angles, circle=False, filter_name='ramp')
    fbp_recon = np.clip(fbp_recon, 0, 1)
    
    print(f"[INFO] FBP reconstruction completed")
    
    # Stage 2: TV-PDHG denoising (Chambolle-Pock)
    # Solve: min_x  0.5 * ||x - x_fbp||^2  +  lam * ||∇x||_{2,1}
    #        s.t.   0 <= x <= 1
    
    # ||∇||^2 <= 8 for 2D forward differences
    norm_grad = np.sqrt(8.0)
    
    # Step sizes (Chambolle-Pock for denoising)
    tau = 1.0 / norm_grad
    sigma = 1.0 / norm_grad
    
    print(f"[INFO] TV-PDHG denoising: lam={lam}, niter={niter_tv}")
    
    # Initialize from FBP
    x = fbp_recon.copy()
    x_bar = x.copy()
    p = np.zeros((2, image_size, image_size), dtype='float64')  # dual variable
    
    for k in range(niter_tv):
        x_old = x.copy()
        
        # Dual update: p = prox_{sigma * g*}(p + sigma * ∇(x_bar))
        p = p + sigma * grad(x_bar)
        # Project onto l2 balls of radius lam
        norms = np.sqrt(p[0]**2 + p[1]**2)
        scale = np.maximum(norms / lam, 1.0)
        p = p / scale[np.newaxis, :, :]
        
        # Primal update: x = prox_{tau * f}(x - tau * (-div(p)))
        x = np.clip((x + tau * div(p) + tau * fbp_recon) / (1.0 + tau), 0, 1)
        
        # Over-relaxation (theta=1)
        x_bar = 2 * x - x_old
        
        if (k + 1) % 50 == 0:
            print(f"  TV iter {k+1:4d}/{niter_tv}")
    
    recon = x.astype('float32')
    print(f"[INFO] Reconstruction range: [{recon.min():.4f}, {recon.max():.4f}]")
    
    return {
        'reconstruction': recon,
        'fbp_reconstruction': fbp_recon.astype('float32')
    }
