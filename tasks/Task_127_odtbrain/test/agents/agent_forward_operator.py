import sys

import os

import matplotlib

matplotlib.use('Agg')

import numpy as np

from scipy.ndimage import rotate

sys.path.insert(0, '/data/yjh/odtbrain_sandbox/repo')

RESULTS_DIR = '/data/yjh/odtbrain_sandbox/results'

os.makedirs(RESULTS_DIR, exist_ok=True)

def forward_operator(f_obj, angles, res, nm):
    """
    Simulate scattered field sinogram using the first-order Born approximation.
    
    For each angle, rotates the object function about the y-axis,
    integrates along the z-axis (propagation direction), and computes
    the Born scattered field at the detector.
    
    The Born scattered field at the detector is approximately:
        u_B(x,y) ~ integral of f(r) * exp(i*km*z) dz
    For weak scattering:
        u_total = u_0 + u_B ≈ 1 + u_B (for unit incident field)
    
    Parameters:
        f_obj: 3D numpy array - object function
        angles: 1D numpy array - projection angles in radians
        res: float - wavelength in pixels
        nm: float - medium refractive index
    
    Returns:
        sino: 3D complex numpy array - scattered field sinogram (A, Ny, Nx)
    """
    N = f_obj.shape[0]
    km_val = 2 * np.pi * nm / res
    num_angles = len(angles)
    
    # Sinogram: (A, Ny, Nx) complex array
    sino = np.zeros((num_angles, N, N), dtype=np.complex128)
    
    for i, angle in enumerate(angles):
        # Rotate the object function about the y-axis (axis=1)
        # ODTbrain convention: rotation about y-axis, in the xz plane
        angle_deg = np.degrees(angle)
        # Rotate f_obj: axes (0,2) means rotation in the xz plane
        f_rotated = rotate(f_obj, angle=-angle_deg, axes=(0, 2),
                           reshape=False, order=1, mode='constant', cval=0)
        
        # Project along z-axis (axis=2 after rotation = propagation direction)
        # The Born scattered field: u_B(x,y) ~ (i/(2*km)) * integral f(r) dz
        projection = np.sum(f_rotated, axis=2).astype(np.complex128)
        
        # Scale by voxel size (1 pixel) and the Born kernel factor
        # u_B = (i / (2*km)) * projection
        u_B = (1j / (2 * km_val)) * projection
        
        # Total field = incident + scattered
        # u_total = u_0 * (1 + u_B/u_0) where u_0 = 1
        sino[i] = 1.0 + u_B
    
    return sino
