import numpy as np


# --- Extracted Dependencies ---

def get_slopes_diffractive_explicit(wfs, phase_in=None):
    """
    Simulates the physical process of the Shack-Hartmann WFS:
    1. Propagate phase to lenslet array.
    2. Form spots (PSFs) for each subaperture via FFT.
    3. Compute Center of Gravity (CoG) of spots to get slopes.
    """
    if phase_in is not None:
        wfs.telescope.src.phase = phase_in

    # A. Get Electric Field at Lenslet Array
    cube_em = wfs.get_lenslet_em_field(wfs.telescope.src.phase)
    
    # B. Form Spots (Intensity = |FFT(E)|^2)
    complex_field = np.fft.fft2(cube_em, axes=[1, 2])
    intensity_spots = np.abs(complex_field)**2
    
    # C. Centroiding (Center of Gravity)
    n_pix = intensity_spots.shape[1]
    x = np.arange(n_pix) - n_pix // 2
    X, Y = np.meshgrid(x, x)
    
    # Compute centroids for valid subapertures
    slopes = np.zeros((wfs.nValidSubaperture, 2))
    valid_idx = 0
    
    for i in range(wfs.nSubap**2):
        if wfs.valid_subapertures_1D[i]:
            I = intensity_spots[i]
            flux = np.sum(I)
            if flux > 0:
                cx = np.sum(I * X) / flux
                cy = np.sum(I * Y) / flux
                slopes[valid_idx, 0] = cx
                slopes[valid_idx, 1] = cy
                valid_idx += 1
                
    slopes_flat = np.concatenate((slopes[:, 0], slopes[:, 1]))
    
    return slopes_flat
