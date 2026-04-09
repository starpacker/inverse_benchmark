import numpy as np

import scipy.ndimage

def symmetric_gaussian_2d(xy, background, height, center_x, center_y, width):
    """
    Explicit mathematical definition of a 2D Symmetric Gaussian.
    f(x,y) = background + height * exp( -2 * ( ((x-cx)/w)^2 + ((y-cy)/w)^2 ) )
    """
    x, y = xy
    g = background + height * np.exp(-2 * (((center_x - x) / width) ** 2 + ((center_y - y) / width) ** 2))
    return g.ravel()

def run_inversion(image, sigma, background_sigma, threshold_factor):
    """
    Performs peak finding and Gaussian fitting (inversion).
    
    Returns: 
        fitted_params: List of optimized gaussian parameters
        estimated_background: The background image estimated during preprocessing
    """
    # --- Step 1: Preprocessing for Peak Finding ---
    smooth_img = scipy.ndimage.gaussian_filter(image, sigma)
    bg_img = scipy.ndimage.gaussian_filter(image, background_sigma)
    dog_img = smooth_img - bg_img
    
    # Thresholding
    threshold = threshold_factor * np.std(dog_img) + np.mean(dog_img)
    mask = dog_img > threshold
    
    # Local Maxima Detection
    neighborhood_size = 3
    local_max = scipy.ndimage.maximum_filter(dog_img, size=neighborhood_size) == dog_img
    
    # Combined mask
    peaks_mask = local_max & mask
    y_peaks, x_peaks = np.where(peaks_mask)
    
    # --- Step 2: Fitting (Levenberg-Marquardt) ---
    fitted_params = []
    r = 5 
    h, w = image.shape
    
    for px, py in zip(x_peaks, y_peaks):
        if px < r or px >= w - r or py < r or py >= h - r:
            continue
            
        # Crop ROI
        roi = image[py-r:py+r+1, px-r:px+r+1]
        y_roi, x_roi = np.mgrid[py-r:py+r+1, px-r:px+r+1]
        
        # Initial Guess: [background, height, center_x, center_y, width]
        p0 = [np.min(roi), np.max(roi) - np.min(roi), px, py, 2.0]
        
        try:
            error_function = lambda p: symmetric_gaussian_2d((x_roi, y_roi), *p) - roi.ravel()
            p_opt, success = scipy.optimize.leastsq(error_function, p0, maxfev=100)
            
            if success in [1, 2, 3, 4]: 
                bg, height, cx, cy, width = p_opt
                # Sanity checks
                if (height > 0) and (0.5 < width < 10.0) and (abs(cx - px) < 2) and (abs(cy - py) < 2):
                    fitted_params.append(p_opt)
        except Exception:
            continue
            
    return fitted_params, bg_img
