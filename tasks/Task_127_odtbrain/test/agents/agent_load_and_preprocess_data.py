import sys

import os

import matplotlib

matplotlib.use('Agg')

import numpy as np

sys.path.insert(0, '/data/yjh/odtbrain_sandbox/repo')

RESULTS_DIR = '/data/yjh/odtbrain_sandbox/results'

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_and_preprocess_data(N, nm, n_sphere, radius, num_angles, res, noise_level, seed=42):
    """
    Generate 3D RI phantom and preprocess data for reconstruction.
    
    Creates a 3D sphere phantom with known refractive index,
    computes the object function, and generates projection angles.
    
    Parameters:
        N: int - grid size (voxels per axis)
        nm: float - medium refractive index
        n_sphere: float - sphere refractive index
        radius: float - sphere radius in voxels
        num_angles: int - number of projection angles
        res: float - wavelength in pixels
        noise_level: float - noise standard deviation relative to signal
        seed: int - random seed for reproducibility
    
    Returns:
        dict containing:
            - phantom_ri: 3D numpy array of refractive index phantom
            - f_obj: 3D numpy array of object function
            - angles: 1D numpy array of projection angles
            - km: float - wave number in medium
            - params: dict of all input parameters
    """
    # Create 3D sphere phantom
    phantom = np.full((N, N, N), nm, dtype=np.float64)
    center = N // 2
    zz, yy, xx = np.mgrid[:N, :N, :N]
    dist = np.sqrt((xx - center)**2 + (yy - center)**2 + (zz - center)**2)
    phantom[dist <= radius] = n_sphere
    
    # Compute object function f from RI
    # f(r) = k_m^2 * [(n(r)/n_m)^2 - 1]
    km = 2 * np.pi * nm / res
    f_obj = km**2 * ((phantom / nm)**2 - 1)
    
    # Generate projection angles
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    
    # Store parameters
    params = {
        'N': N,
        'nm': nm,
        'n_sphere': n_sphere,
        'radius': radius,
        'num_angles': num_angles,
        'res': res,
        'noise_level': noise_level,
        'seed': seed,
        'km': km
    }
    
    return {
        'phantom_ri': phantom,
        'f_obj': f_obj,
        'angles': angles,
        'km': km,
        'params': params
    }
