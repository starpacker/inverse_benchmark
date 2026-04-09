import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.fft import fft2, ifft2, fftshift, ifftshift

from scipy.ndimage import gaussian_filter, binary_dilation

def run_inversion(intensity_noisy, support_init, n_hio, n_er, beta,
                  shrinkwrap_interval, shrinkwrap_sigma, shrinkwrap_threshold,
                  n_starts, seed):
    """
    Full HIO+ER phase retrieval with shrink-wrap support and multi-start.
    
    Args:
        intensity_noisy: Measured diffraction intensity
        support_init: Initial support estimate
        n_hio: Number of HIO iterations
        n_er: Number of ER iterations
        beta: HIO feedback parameter
        shrinkwrap_interval: Interval for support updates
        shrinkwrap_sigma: Gaussian blur sigma for shrink-wrap
        shrinkwrap_threshold: Threshold for support estimation
        n_starts: Number of random starts
        seed: Random seed
        
    Returns:
        obj_rec: Reconstructed complex object
        support_final: Final support estimate
        errors: List of R-factor errors during convergence
    """
    sqrt_intensity = np.sqrt(intensity_noisy)
    det_size = intensity_noisy.shape[0]
    
    best_obj_rec = None
    best_err = float('inf')
    best_errors = []
    best_support = None
    
    for start_i in range(n_starts):
        rng_start = np.random.default_rng(seed + start_i)
        
        # Random initial guess
        phase_init = 2 * np.pi * rng_start.random((det_size, det_size))
        obj_est = support_init.astype(complex) * np.exp(1j * phase_init)
        
        support = support_init.copy()
        errors = []
        
        print(f"  Start {start_i+1}/{n_starts}: Phase retrieval: {n_hio} HIO + {n_er} ER iterations")
        
        # HIO phase
        for it in range(n_hio):
            # HIO iteration
            # Fourier constraint: replace amplitude
            F_est = fftshift(fft2(ifftshift(obj_est)))
            phase_est = np.angle(F_est)
            F_constrained = sqrt_intensity * np.exp(1j * phase_est)
            obj_new = fftshift(ifft2(ifftshift(F_constrained)))
            
            # Real-space constraint (HIO)
            obj_out = obj_est.copy()
            outside_support = ~support
            
            # Inside support: keep new estimate
            obj_out[support] = obj_new[support]
            # Outside support: HIO feedback
            obj_out[outside_support] = obj_est[outside_support] - beta * obj_new[outside_support]
            
            obj_est = obj_out
            
            # Shrink-wrap support update
            if (it + 1) % shrinkwrap_interval == 0:
                # Shrink-wrap support estimation: blur amplitude, threshold
                amp = np.abs(obj_est)
                amp_smooth = gaussian_filter(amp, sigma=shrinkwrap_sigma)
                amp_smooth = amp_smooth / max(amp_smooth.max(), 1e-12)
                support = amp_smooth > shrinkwrap_threshold
                # Dilate slightly for stability
                support = binary_dilation(support, iterations=1)
                print(f"    HIO iter {it+1:4d}: support pixels = {support.sum()}")
            
            # Error metric
            F_est = fftshift(fft2(ifftshift(obj_est)))
            err = np.sqrt(np.mean((np.abs(F_est) - sqrt_intensity)**2))
            errors.append(err)
            
            if (it + 1) % 50 == 0:
                print(f"    HIO iter {it+1:4d}: R-factor = {err:.6f}")
        
        # ER phase (refinement)
        for it in range(n_er):
            # ER iteration
            F_est = fftshift(fft2(ifftshift(obj_est)))
            phase_est = np.angle(F_est)
            F_constrained = sqrt_intensity * np.exp(1j * phase_est)
            obj_new = fftshift(ifft2(ifftshift(F_constrained)))
            
            # Hard support constraint
            obj_new[~support] = 0
            obj_est = obj_new
            
            if (it + 1) % 25 == 0:
                F_est = fftshift(fft2(ifftshift(obj_est)))
                err = np.sqrt(np.mean((np.abs(F_est) - sqrt_intensity)**2))
                errors.append(err)
                print(f"    ER  iter {it+1:4d}: R-factor = {err:.6f}")
        
        final_err = errors[-1] if errors else float('inf')
        print(f"  Start {start_i+1}/{n_starts}: final R-factor = {final_err:.6f}")
        
        if final_err < best_err:
            best_err = final_err
            best_obj_rec = obj_est.copy()
            best_errors = errors.copy()
            best_support = support.copy()
    
    print(f"  Best start: R-factor = {best_err:.6f}")
    
    return best_obj_rec, best_support, best_errors
