import sys

import os

import matplotlib

matplotlib.use('Agg')

import numpy as np

sys.path.insert(0, '/data/yjh/odtbrain_sandbox/repo')

import odtbrain as odt

RESULTS_DIR = '/data/yjh/odtbrain_sandbox/results'

os.makedirs(RESULTS_DIR, exist_ok=True)

def run_inversion(sino_noisy, angles, res, nm, lD, phantom_shape):
    """
    Reconstruct 3D refractive index using Rytov backpropagation.
    
    Converts sinogram to Rytov phase, applies 3D backpropagation,
    converts object function to refractive index, and aligns
    the reconstruction to the phantom shape.
    
    Parameters:
        sino_noisy: 3D complex numpy array - noisy scattered field sinogram
        angles: 1D numpy array - projection angles in radians
        res: float - wavelength in pixels
        nm: float - medium refractive index
        lD: float - detector distance from rotation center
        phantom_shape: tuple - shape of the original phantom for alignment
    
    Returns:
        dict containing:
            - ri_recon_aligned: 3D numpy array - aligned reconstructed RI
            - f_recon: 3D numpy array - reconstructed object function
            - sino_rytov: 3D numpy array - Rytov-transformed sinogram
    """
    # Convert sinogram to Rytov phase
    sino_rytov = odt.sinogram_as_rytov(sino_noisy)
    
    # 3D Backpropagation
    f_recon = odt.backpropagate_3d(
        uSin=sino_rytov,
        angles=angles,
        res=res,
        nm=nm,
        lD=lD,
        padfac=1.75,
        padding=(True, True),
        padval="edge",
        onlyreal=False,
        intp_order=2,
        save_memory=True,
        num_cores=1,
    )
    
    # Convert object function to refractive index
    ri_recon = odt.odt_to_ri(f_recon, res=res, nm=nm)
    
    # Align volumes (reconstruction may be different size due to padding)
    recon_shape = ri_recon.shape
    
    if recon_shape != phantom_shape:
        # Center-crop the reconstruction
        ri_aligned = np.full(phantom_shape, nm, dtype=np.complex128)
        slices_src = []
        slices_dst = []
        for i in range(3):
            r = recon_shape[i]
            p = phantom_shape[i]
            if r >= p:
                start = (r - p) // 2
                slices_src.append(slice(start, start + p))
                slices_dst.append(slice(0, p))
            else:
                start = (p - r) // 2
                slices_src.append(slice(0, r))
                slices_dst.append(slice(start, start + r))
        ri_aligned[slices_dst[0], slices_dst[1], slices_dst[2]] = \
            ri_recon[slices_src[0], slices_src[1], slices_src[2]]
        ri_recon_aligned = ri_aligned.real
    else:
        ri_recon_aligned = ri_recon.real
    
    return {
        'ri_recon_aligned': ri_recon_aligned,
        'f_recon': f_recon,
        'sino_rytov': sino_rytov
    }
