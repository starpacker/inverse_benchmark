import math

import numpy as np

import scipy.signal

def forward_operator(field, wavelength, pixel_size, propagation_distance, precision="single"):
    """
    Propagates a complex field by a specific distance using the Angular Spectrum Method.
    This acts as the forward model: Object Plane -> Hologram Plane.
    
    Args:
        field (np.ndarray): Complex field at source plane.
        wavelength (float): Wavelength of light (meters).
        pixel_size (float): Pixel pitch (meters).
        propagation_distance (float): Distance to propagate (meters).
        precision (str): 'single' or 'double'.
    
    Returns:
        prop_field (np.ndarray): Complex field at destination plane.
    """
    if precision == "double":
        dtype_c = np.complex128
    else:
        dtype_c = np.complex64

    # Ensure input is complex
    field = field.astype(dtype_c)
    
    grid_height, grid_width = field.shape
    
    # 1. Coordinate Grids for Transfer Function
    # Physical size
    width_phys = grid_width * pixel_size
    height_phys = grid_height * pixel_size
    
    # Frequency coordinates (fx, fy)
    # Using scipy.fft.fftfreq ensures correct ordering (0, positive, negative)
    fx = scipy.fft.fftfreq(grid_width, d=pixel_size)
    fy = scipy.fft.fftfreq(grid_height, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy)
    
    # 2. Angular Spectrum Transfer Function (H)
    # H = exp(j * 2*pi * z/lambda * sqrt(1 - (lambda*fx)^2 - (lambda*fy)^2))
    # We use the k-vector formulation: k = 2*pi/lambda
    # kz = sqrt(k^2 - kx^2 - ky^2) = 2*pi * sqrt(1/lambda^2 - fx^2 - fy^2)
    
    squared_sum = FX**2 + FY**2
    inv_lambda_sq = 1.0 / (wavelength**2)
    
    # Handle evanescent waves (where argument to sqrt is negative)
    # We set them to zero or handle via complex sqrt. Standard ASM filters them out.
    argument = inv_lambda_sq - squared_sum
    
    # Mask for evanescent waves
    mask = argument >= 0
    root_val = np.sqrt(np.maximum(argument, 0)) # maximum to avoid warning on negative
    
    # Phase factor
    # For propagation by distance z: exp(i * 2*pi * z * sqrt(...))
    # Note on sign convention: If using exp(-iwt), forward spatial propagation is exp(ikz).
    # The original code had exp(-1j * ...), which suggests exp(iwt) convention or similar. 
    # We will stick to the logic derived from the provided legacy code:
    # Legacy: prop_corner = np.exp((-1j * 2 * math.pi * depth * sqrt_val / wavelength))
    # where sqrt_val was sqrt(1 - alpha^2 - beta^2). alpha = lambda*fx.
    # So sqrt_val/lambda = sqrt(1/lambda^2 - fx^2). Matches our root_val.
    
    phase = -1j * 2 * math.pi * propagation_distance * root_val
    H = np.exp(phase)
    H[~mask] = 0 # Filter evanescent
    
    H = H.astype(dtype_c)

    # 3. FFT -> Multiply -> IFFT
    F_field = scipy.fft.fft2(field)
    F_prop = F_field * H
    prop_field = scipy.fft.ifft2(F_prop)
    
    return prop_field

def run_inversion(measurements, wavelength, pixel_size, depth, precision="single"):
    """
    Performs the inversion (reconstruction) of the hologram.
    For inline holography, this typically involves back-propagating the 
    background-subtracted interference pattern to the object plane.
    
    Args:
        measurements (np.ndarray): Preprocessed hologram (complex field approximation).
        wavelength (float): Light wavelength.
        pixel_size (float): Pixel size.
        depth (float): Reconstruction depth (z-distance).
        precision (str): 'single' or 'double'.
        
    Returns:
        reconstruction (np.ndarray): The reconstructed complex field at the object plane.
    """
    # In simple inline holography (Gabor), the inversion is often approximated
    # by back-propagation (adjoint of the forward operator).
    # Forward: z. Inverse: -z.
    
    # We use the forward_operator function but with negative distance.
    reconstruction = forward_operator(
        field=measurements,
        wavelength=wavelength,
        pixel_size=pixel_size,
        propagation_distance=-depth, # Negative for back-propagation
        precision=precision
    )
    
    return reconstruction
