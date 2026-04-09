import numpy as np

import matplotlib

matplotlib.use('Agg')

from scipy.ndimage import gaussian_filter

def run_inversion(data, clean_gain=0.05, clean_niter=10000, clean_thresh=0.005, beam_fwhm=3.0):
    """
    Run CLEAN deconvolution and Wiener filtering to reconstruct the image.
    
    Parameters
    ----------
    data : dict - output from load_and_preprocess_data
    clean_gain : float - CLEAN loop gain
    clean_niter : int - max CLEAN iterations
    clean_thresh : float - CLEAN threshold (fraction of peak)
    beam_fwhm : float - restoring beam FWHM in pixels
    
    Returns
    -------
    result : dict containing:
        - clean_image: CLEAN reconstruction
        - wiener_image: Wiener filtered reconstruction
        - final_image: best reconstruction
        - dirty_image: dirty image
        - dirty_beam: dirty beam (PSF)
        - clean_components: list of (y, x, flux) tuples
        - residual: CLEAN residual image
        - method_used: string indicating which method was chosen
    """
    u = data['u']
    v = data['v']
    visibilities = data['visibilities']
    valid = data['valid']
    n = data['config']['n']
    
    # Make dirty image by gridding visibilities and inverse FFT
    uv_grid = np.zeros((n, n), dtype=complex)
    weight_grid = np.zeros((n, n))
    
    u_pix = np.round(u + n // 2).astype(int)
    v_pix = np.round(v + n // 2).astype(int)
    
    for i in range(len(u)):
        if valid[i]:
            ui, vi = u_pix[i], v_pix[i]
            if 0 <= ui < n and 0 <= vi < n:
                uv_grid[vi, ui] += visibilities[i]
                weight_grid[vi, ui] += 1.0
    
    # Natural weighting
    mask = weight_grid > 0
    uv_grid[mask] /= weight_grid[mask]
    
    # Sampling function (for PSF)
    sampling = (weight_grid > 0).astype(float)
    
    # Dirty image = IFFT of gridded visibilities
    dirty_image = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(uv_grid))))
    
    # Dirty beam = IFFT of sampling function
    dirty_beam = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(sampling))))
    
    # Normalize dirty beam to peak = 1
    dirty_beam /= np.max(dirty_beam)
    
    # Högbom CLEAN algorithm
    residual = dirty_image.copy()
    clean_components = []
    
    # Absolute threshold
    abs_threshold = clean_thresh * np.max(np.abs(dirty_image))
    
    # Pre-compute beam FFT for fast subtraction
    beam_fft = np.fft.fft2(np.fft.ifftshift(dirty_beam))
    
    for iteration in range(clean_niter):
        # Find peak in residual
        peak_idx = np.argmax(np.abs(residual))
        peak_y, peak_x = np.unravel_index(peak_idx, residual.shape)
        peak_val = residual[peak_y, peak_x]
        
        if np.abs(peak_val) < abs_threshold:
            print(f"  CLEAN converged at iteration {iteration}, "
                  f"residual peak = {np.abs(peak_val):.4e}")
            break
        
        # Subtract shifted dirty beam using shift theorem
        flux = clean_gain * peak_val
        clean_components.append((peak_y, peak_x, flux))
        
        # Create delta function at peak location
        delta = np.zeros((n, n))
        delta[peak_y, peak_x] = flux
        
        # Convolved beam at this location
        delta_fft = np.fft.fft2(delta)
        subtraction = np.real(np.fft.ifft2(delta_fft * beam_fft))
        
        residual -= subtraction
        
        if (iteration + 1) % 1000 == 0:
            print(f"  CLEAN iter {iteration+1}: peak = {np.abs(peak_val):.4e}, "
                  f"components = {len(clean_components)}")
    
    # Restore CLEAN image
    cc_image = np.zeros((n, n))
    for (y, x, flux) in clean_components:
        if 0 <= y < n and 0 <= x < n:
            cc_image[y, x] += flux
    
    # Convolve with Gaussian restoring beam
    sigma = beam_fwhm / (2 * np.sqrt(2 * np.log(2)))
    restored = gaussian_filter(cc_image, sigma=sigma)
    
    # Add residual
    clean_image = restored + residual
    
    # Wiener filtering
    sampling_fft = np.fft.fft2(np.fft.ifftshift(dirty_beam))
    vis_grid_fft = np.fft.fft2(np.fft.ifftshift(dirty_image))
    
    wiener_lambda = 0.01 * np.max(np.abs(sampling_fft))**2
    wiener_filter = np.conj(sampling_fft) / (np.abs(sampling_fft)**2 + wiener_lambda)
    wiener_image = np.real(np.fft.fftshift(np.fft.ifft2(wiener_filter * vis_grid_fft)))
    wiener_image = np.maximum(wiener_image, 0)
    
    # Normalize and choose best result
    clean_image = np.maximum(clean_image, 0)
    model = data['model']
    
    if clean_image.max() > 0:
        clean_image_norm = clean_image * model.max() / clean_image.max()
    else:
        clean_image_norm = clean_image.copy()
    
    if wiener_image.max() > 0:
        wiener_image_norm = wiener_image * model.max() / wiener_image.max()
    else:
        wiener_image_norm = wiener_image.copy()
    
    # Compute PSNR for comparison
    def compute_psnr_internal(ref, test):
        data_range = ref.max() - ref.min()
        mse = np.mean((ref.astype(float) - test.astype(float))**2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10(data_range**2 / mse)
    
    psnr_clean = compute_psnr_internal(model, clean_image_norm)
    psnr_wiener = compute_psnr_internal(model, wiener_image_norm)
    
    print(f"  CLEAN PSNR: {psnr_clean:.2f} dB, Wiener PSNR: {psnr_wiener:.2f} dB")
    
    if psnr_wiener > psnr_clean:
        print("  → Using Wiener-filtered image (better quality)")
        final_image = wiener_image_norm
        method_used = "Wiener_filter"
    else:
        print("  → Using CLEAN image")
        final_image = clean_image_norm
        method_used = "Hogbom_CLEAN"
    
    result = {
        'clean_image': clean_image_norm,
        'wiener_image': wiener_image_norm,
        'final_image': final_image,
        'dirty_image': dirty_image,
        'dirty_beam': dirty_beam,
        'clean_components': clean_components,
        'residual': residual,
        'method_used': method_used,
        'psnr_clean': psnr_clean,
        'psnr_wiener': psnr_wiener
    }
    
    return result
