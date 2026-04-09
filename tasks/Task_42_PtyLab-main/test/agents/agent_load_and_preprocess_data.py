import numpy as np

import h5py

import scipy.fft

from scipy.ndimage import shift, gaussian_filter

import os

def fft2c(x):
    """Centered 2D FFT."""
    return scipy.fft.fftshift(scipy.fft.fft2(scipy.fft.ifftshift(x)))

def generate_initial_probe(Np, dxo, diameter):
    """Generates a soft-edged disk probe."""
    Y, X = np.meshgrid(np.arange(Np), np.arange(Np), indexing='ij')
    X = X - Np // 2
    Y = Y - Np // 2
    R = np.sqrt(X**2 + Y**2)
    
    if diameter is not None:
         fwhm_pix = diameter / dxo
         radius_pix = fwhm_pix / 2.0
    else:
         radius_pix = Np / 8 
         
    probe = np.zeros((Np, Np), dtype=np.complex128)
    probe[R <= radius_pix] = 1.0
    # Soften edges slightly
    probe = gaussian_filter(probe.real, sigma=2.0) + 0j
    return probe

def load_and_preprocess_data(filepath):
    """
    Loads HDF5 data, calculates geometry, and prepares initial arrays.
    
    Returns:
        dict: A dictionary containing all necessary data and parameters.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"Loading data from {filepath}...")
    
    with h5py.File(filepath, 'r') as f:
        ptychogram = f['ptychogram'][:]
        encoder = f['encoder'][:]  # Positions in meters
        dxd = f['dxd'][0]          # Detector pixel size
        Nd = int(f['Nd'][0])       # Detector size
        No = int(f['No'][0])       # Object size
        zo = f['zo'][0]            # Distance
        wavelength = f['wavelength'][0]
        
        entrance_pupil = f['entrancePupilDiameter'][0] if 'entrancePupilDiameter' in f else None
        gt_object = f['object'][:] if 'object' in f else None
    
    # Geometry calculations
    Ld = Nd * dxd
    dxo = wavelength * zo / Ld  # Object pixel size
    
    # Calculate positions in pixels
    # Center positions relative to the grid
    pos_relative_pix = np.round(encoder / dxo).astype(int)
    
    # Map to array indices. Probe size Np is usually set to Detector size Nd.
    Np = Nd
    
    # Calculate top-left indices for crop extraction
    # Assumes object is centered at No//2
    positions = pos_relative_pix + No // 2 - Np // 2
    
    # Initialize Object (Flat start)
    initial_object = np.ones((No, No), dtype=np.complex128)
    
    # Initialize Probe
    initial_probe = generate_initial_probe(Np, dxo, entrance_pupil)
    
    # Energy scaling for the probe
    # Scale such that mean intensity of simulation matches mean intensity of data
    test_wave = fft2c(initial_probe)
    test_intensity = np.abs(test_wave)**2
    scale_factor = np.sqrt(np.mean(ptychogram) / (np.mean(test_intensity) + 1e-10))
    initial_probe *= scale_factor
    
    print(f"  Detector: {Nd}x{Nd}, Object: {No}x{No}")
    print(f"  Pixel size (obj): {dxo:.2e} m")
    print(f"  Probe scaled by: {scale_factor:.2f}")

    data_container = {
        'ptychogram': ptychogram,  # The diffraction patterns (intensity)
        'positions': positions,    # Top-left indices (N_pos, 2)
        'Nd': Nd,
        'No': No,
        'Np': Np,
        'initial_object': initial_object,
        'initial_probe': initial_probe,
        'ground_truth_object': gt_object
    }
    
    return data_container
