import numpy as np

import matplotlib

matplotlib.use('Agg')

from skimage.transform import radon, iradon, resize

from skimage.restoration import denoise_tv_chambolle

def run_inversion(sinogram_sparse, angles_sparse, size, n_iter=200, tv_weight=0.008, step_size=None):
    """
    Run FISTA-TV iterative reconstruction for sparse-view CT.
    
    Solves: min_x  0.5 * ||A*x - y||^2 + lambda * TV(x)
    where A is the Radon transform at sparse angles, y is the sparse sinogram.
    
    Args:
        sinogram_sparse: Sparse sinogram (n_detectors x n_sparse_angles)
        angles_sparse: Sparse angle array
        size: Output image size
        n_iter: Number of FISTA iterations
        tv_weight: TV regularization weight
        step_size: Step size (if None, estimated via power iteration)
        
    Returns:
        tv_recon: TV-regularized reconstruction (size x size)
        fbp_sparse: FBP reconstruction from sparse data (size x size)
        iteration_info: Dictionary with iteration information
    """
    
    def radon_forward(image, angles):
        """Forward Radon transform."""
        return radon(image, theta=angles, circle=True)
    
    def radon_adjoint(sinogram, angles, output_size):
        """Adjoint (backprojection without filter) operator."""
        return iradon(sinogram, theta=angles, circle=True,
                      output_size=output_size, filter_name=None)
    
    def fbp_reconstruction(sinogram, angles, output_size):
        """Filtered back-projection reconstruction."""
        recon = iradon(sinogram, theta=angles, circle=True,
                       output_size=output_size, filter_name='ramp')
        return recon
    
    def tv_proximal(x, weight):
        """TV proximal operator using Chambolle's algorithm."""
        x_clipped = np.clip(x, 0, None)
        denoised = denoise_tv_chambolle(x_clipped, weight=weight)
        return denoised
    
    # Estimate step size from operator norm (Lipschitz constant) via power iteration
    if step_size is None:
        x_test = np.random.randn(size, size)
        for _ in range(5):
            y_test = radon_forward(x_test, angles_sparse)
            x_test = radon_adjoint(y_test, angles_sparse, size)
            norm_est = np.sqrt(np.sum(x_test ** 2))
            x_test = x_test / norm_est
        step_size = 1.0 / norm_est
        print(f"  Estimated step size: {step_size:.6f} (L={norm_est:.1f})")
    
    # Initialize with FBP
    fbp_sparse = fbp_reconstruction(sinogram_sparse, angles_sparse, size)
    x = np.clip(fbp_sparse, 0, None)
    x_prev = x.copy()
    t = 1.0
    
    iteration_info = {
        'data_fit_history': [],
        'step_size': step_size,
        'norm_est': norm_est if step_size is None else None
    }
    
    print(f"  Running FISTA-TV: {n_iter} iterations, tv_weight={tv_weight}")
    for k in range(n_iter):
        # Momentum (FISTA)
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        momentum = (t - 1) / t_new
        z = x + momentum * (x - x_prev)
        
        # Gradient step: gradient of 0.5*||Ax - y||^2 = A^T(Ax - y)
        residual = radon_forward(z, angles_sparse) - sinogram_sparse
        gradient = radon_adjoint(residual, angles_sparse, size)
        z_grad = z - step_size * gradient
        
        # TV proximal step
        x_new = tv_proximal(z_grad, weight=tv_weight * step_size)
        
        # Update
        x_prev = x.copy()
        x = x_new
        t = t_new
        
        if (k + 1) % 30 == 0 or k == 0:
            data_fit = 0.5 * np.sum(residual ** 2)
            iteration_info['data_fit_history'].append((k + 1, data_fit))
            print(f"    Iter {k+1:3d}: data_fit = {data_fit:.2f}")
    
    tv_recon = np.clip(x, 0, None)
    
    return tv_recon, fbp_sparse, iteration_info
