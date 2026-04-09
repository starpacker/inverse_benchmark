import numpy as np

import matplotlib

matplotlib.use('Agg')

from skimage.transform import resize, radon, iradon

from skimage.restoration import denoise_tv_chambolle

def run_inversion(sino_noisy, angles_sparse, image_size, n_outer_iter, n_tv_iter, tv_weight, dc_step_size):
    """
    Perform diffusion-style iterative CT reconstruction.
    
    This implements:
    1. Initialize with FBP
    2. For each iteration:
       a. TV denoising (acts as learned prior/denoiser proxy)
       b. Data consistency (gradient step to match measurements)
    3. Return refined reconstruction
    
    Parameters:
    -----------
    sino_noisy : ndarray
        Noisy sparse-view sinogram
    angles_sparse : ndarray
        Array of sparse projection angles
    image_size : int
        Size of the reconstruction (image_size x image_size)
    n_outer_iter : int
        Number of outer iterations
    n_tv_iter : int
        Number of TV denoising sub-iterations
    tv_weight : float
        TV regularization weight
    dc_step_size : float
        Data consistency step size
        
    Returns:
    --------
    recon_diffusion : ndarray
        Final diffusion-style reconstruction
    recon_fbp : ndarray
        Initial FBP reconstruction (baseline)
    """
    # FBP reconstruction helper
    def fbp_reconstruct(sinogram, angles, size):
        """Filtered Back Projection reconstruction."""
        recon = iradon(sinogram, theta=angles, circle=True, filter_name='ramp')
        if recon.shape[0] != size:
            recon = resize(recon, (size, size), anti_aliasing=True)
        return recon
    
    # TV denoising helper
    def tv_denoise(image, weight, n_iter):
        """Total Variation denoising using Chambolle's projection algorithm."""
        return denoise_tv_chambolle(image, weight=weight, max_num_iter=n_iter)
    
    # Data consistency step helper
    def data_consistency_step(image, sinogram, angles, step_size):
        """
        Data consistency: project current estimate, compute residual,
        back-project residual to enforce measurement consistency.
        """
        # Forward project current estimate
        sino_est = radon(image, theta=angles, circle=True)
        
        # Residual in sinogram domain
        residual_sino = sinogram - sino_est
        
        # Back-project residual WITHOUT filter (gradient of data fidelity)
        correction = iradon(residual_sino, theta=angles, circle=True, filter_name=None)
        
        # Resize if needed
        if correction.shape != image.shape:
            correction = resize(correction, image.shape, anti_aliasing=True)
        
        # Normalize correction
        if np.max(np.abs(correction)) > 0:
            correction = correction / np.max(np.abs(correction)) * step_size
        
        # Apply correction
        return image + correction
    
    # Initialize with FBP
    x = fbp_reconstruct(sino_noisy, angles_sparse, image_size)
    x_fbp = x.copy()
    
    print(f"[RECON] Starting diffusion-style iterative refinement...")
    print(f"  Config: {n_outer_iter} outer iters, TV weight={tv_weight}, DC step={dc_step_size}")
    
    # Adaptive TV weight schedule (decrease over iterations, like noise schedule)
    tv_schedule = np.linspace(tv_weight * 2, tv_weight * 0.5, n_outer_iter)
    dc_schedule = np.linspace(dc_step_size * 0.3, dc_step_size * 0.8, n_outer_iter)
    
    for i in range(n_outer_iter):
        # Step 1: Denoise (prior/score function proxy)
        x_denoised = tv_denoise(x, weight=tv_schedule[i], n_iter=n_tv_iter)
        
        # Step 2: Data consistency
        x = data_consistency_step(x_denoised, sino_noisy, angles_sparse, step_size=dc_schedule[i])
        
        # Clip to valid range
        x = np.clip(x, 0, 1)
        
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Iter {i+1}/{n_outer_iter}: TV_w={tv_schedule[i]:.5f}, range=[{x.min():.3f}, {x.max():.3f}]")
    
    return x, x_fbp
