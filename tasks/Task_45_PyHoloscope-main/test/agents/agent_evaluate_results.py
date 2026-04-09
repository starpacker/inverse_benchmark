import math

import numpy as np

import scipy.signal

from skimage.metrics import peak_signal_noise_ratio as psnr

from skimage.metrics import structural_similarity as ssim

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

def evaluate_results(reconstruction, preprocessed_input, wavelength, pixel_size, depth):
    """
    Evaluates the consistency of the reconstruction.
    Since ground truth is often unavailable in experimental data, we perform
    'self-consistency' checks:
    1. Propagate the reconstruction BACK to the hologram plane (Forward Model).
    2. Compare the amplitude of this simulation with the input preprocessed amplitude.
    
    Args:
        reconstruction (np.ndarray): Object plane field.
        preprocessed_input (np.ndarray): Hologram plane field (input to inversion).
        wavelength, pixel_size, depth: Physical parameters.
        
    Returns:
        metrics (dict): Dictionary containing PSNR and SSIM.
        simulated_holo (np.ndarray): The forward-projected reconstruction.
    """
    # 1. Forward propagate the result back to the sensor plane
    # The reconstruction is at z=0 (relative to object). Sensor is at z=depth.
    # Note: run_inversion went -depth. To check consistency, we go +depth.
    simulated_holo_field = forward_operator(
        field=reconstruction,
        wavelength=wavelength,
        pixel_size=pixel_size,
        propagation_distance=depth,
        precision="single"
    )
    
    # 2. Extract Amplitudes for Comparison
    # We compare |Forward(Recon)| vs |Input|.
    rec_amp = np.abs(simulated_holo_field)
    inp_amp = np.abs(preprocessed_input)
    
    # 3. Normalize to [0, 1] range for metric calculation
    def normalize(x):
        xmin, xmax = x.min(), x.max()
        if xmax - xmin < 1e-9: return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin)
        
    ref_norm = normalize(inp_amp)
    test_norm = normalize(rec_amp)
    
    # 4. Calculate Metrics
    # data_range is 1.0 because we normalized
    val_psnr = psnr(ref_norm, test_norm, data_range=1.0)
    val_ssim = ssim(ref_norm, test_norm, data_range=1.0)
    
    return {"psnr": val_psnr, "ssim": val_ssim}, simulated_holo_field
