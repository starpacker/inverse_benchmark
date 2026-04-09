# -*- coding: utf-8 -*-
"""
Refactored Holography Pipeline
Components:
1. Data Loading & Preprocessing
2. Forward Operator (Propagator)
3. Inversion (Refocusing)
4. Evaluation
"""

import math
import time
import warnings
from pathlib import Path
import numpy as np
import scipy.fft
import scipy.signal
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# =============================================================================
# 1. LOAD AND PREPROCESS DATA
# =============================================================================

def load_and_preprocess_data(holo_path, back_path=None, downsample=1.0, window_func=None, precision="single"):
    """
    Loads hologram and background images, applies background subtraction,
    normalization (optional), windowing, and downsampling.
    
    Returns:
        processed_img (np.ndarray): The complex or real preprocessed field.
        raw_holo (np.ndarray): The raw loaded hologram for visualization.
        raw_back (np.ndarray): The raw loaded background (or None).
    """
    # 1. Define Precision
    if precision == "double":
        dtype_complex = "complex128"
        dtype_real = "float64"
    else:
        dtype_complex = "complex64"
        dtype_real = "float32"

    # 2. Helper: Load Image
    def _load_image(filename):
        im = Image.open(filename)
        # Handle multi-frame tiff if necessary, though typical for inline is single frame
        if hasattr(im, 'n_frames') and im.n_frames > 1:
            im.seek(0) # Just take first frame for this simple pipeline
        return np.array(im)

    # 3. Load Data
    raw_holo = _load_image(holo_path)
    img = raw_holo.astype(dtype_real)
    
    raw_back = None
    background = None
    if back_path:
        raw_back = _load_image(back_path)
        background = raw_back.astype(dtype_real)

    # 4. Remove Background
    # Logic: If complex, (Amp - sqrt(Back)) * exp(j*Phase). If real, Img - Back.
    if background is not None:
        if background.shape != img.shape:
             # Basic resize if shapes mismatch (robustness)
             background = cv.resize(background, (img.shape[1], img.shape[0]))
        
        if np.iscomplexobj(img):
            img_amp = np.abs(img)
            img_phase = np.angle(img)
            # Ensure non-negative inside sqrt for background
            bg_sqrt = np.sqrt(np.maximum(background, 0))
            img = (img_amp - bg_sqrt) * np.exp(1j * img_phase)
        else:
            img = img - background

    # 5. Apply Window
    # Create a window if one isn't passed but a string name might be (not implemented here, assuming array passed)
    if window_func is not None:
        if window_func.shape != img.shape:
            # Resize window to match image
            window_resized = cv.resize(window_func, (img.shape[1], img.shape[0]))
        else:
            window_resized = window_func
            
        if np.iscomplexobj(img):
            img.real *= window_resized
            img.imag *= window_resized
        else:
            img *= window_resized

    # 6. Downsample
    if downsample != 1.0:
        new_w = int(img.shape[1] / downsample / 2) * 2
        new_h = int(img.shape[0] / downsample / 2) * 2
        if np.iscomplexobj(img):
            # Resize real and imag separately
            r = cv.resize(img.real, (new_w, new_h))
            i = cv.resize(img.imag, (new_w, new_h))
            img = r + 1j*i
        else:
            img = cv.resize(img, (new_w, new_h))

    # Cast final result to correct complex type for propagation
    img = img.astype(dtype_complex)
    
    return img, raw_holo, raw_back

# =============================================================================
# 2. FORWARD OPERATOR
# =============================================================================

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

# =============================================================================
# 3. RUN INVERSION
# =============================================================================

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

# =============================================================================
# 4. EVALUATE RESULTS
# =============================================================================

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

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("Initializing PyHoloscope Refactored Pipeline...")
    
    # --- Configuration ---
    # Try to locate test files dynamically based on script location
    script_dir = Path(__file__).parent.absolute()
    # Common paths for the provided context
    potential_paths = [
        script_dir / "test/integration_tests/test data",
        script_dir / "../test/integration_tests/test data",
        script_dir # If in same folder
    ]
    
    holo_file = None
    back_file = None
    
    for p in potential_paths:
        h = p / "inline_example_holo.tif"
        b = p / "inline_example_back.tif"
        if h.exists() and b.exists():
            holo_file = h
            back_file = b
            break
            
    if holo_file is None:
        # Create dummy data if files don't exist (strictly for runnable code requirement)
        print("Warning: Test files not found. Generating synthetic data for demonstration.")
        holo_file = "dummy_holo.tif"
        back_file = "dummy_back.tif"
        # Generate synthetic zone plate
        N = 512
        x = np.linspace(-1, 1, N)
        X, Y = np.meshgrid(x, x)
        R2 = X**2 + Y**2
        synth_holo = 0.5 + 0.5 * np.cos(100 * R2)
        synth_back = np.ones_like(synth_holo) * 0.5
        Image.fromarray((synth_holo*255).astype(np.uint8)).save(holo_file)
        Image.fromarray((synth_back*255).astype(np.uint8)).save(back_file)

    # Physical Parameters
    WAVELENGTH = 630e-9
    PIXEL_SIZE = 1e-6
    DEPTH = 0.0130
    PRECISION = "single"
    
    # 1. Load and Preprocess
    print(f"Loading data from {holo_file}")
    preprocessed_data, raw_holo, _ = load_and_preprocess_data(
        str(holo_file), 
        str(back_file), 
        precision=PRECISION
    )
    print(f"Data Preprocessed. Shape: {preprocessed_data.shape}, Type: {preprocessed_data.dtype}")

    # 2. Forward Operator Test (Sanity Check - not strictly part of inversion flow but good practice)
    # We just ensure it runs
    _ = forward_operator(preprocessed_data, WAVELENGTH, PIXEL_SIZE, 0.0, precision=PRECISION)

    # 3. Run Inversion
    print("Running Inversion (Back-propagation)...")
    t0 = time.time()
    reconstructed_field = run_inversion(
        measurements=preprocessed_data,
        wavelength=WAVELENGTH,
        pixel_size=PIXEL_SIZE,
        depth=DEPTH,
        precision=PRECISION
    )
    print(f"Inversion complete in {time.time() - t0:.4f}s")
    
    # 4. Evaluate
    print("Evaluating Results...")
    metrics, simulated_holo = evaluate_results(
        reconstruction=reconstructed_field,
        preprocessed_input=preprocessed_data,
        wavelength=WAVELENGTH,
        pixel_size=PIXEL_SIZE,
        depth=DEPTH
    )
    
    print(f"Self-Consistency Metrics -> PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
    
    # Visualization (Save to file)
    rec_amp = np.abs(reconstructed_field)
    sim_amp = np.abs(simulated_holo)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Preprocessed Input")
    plt.imshow(np.abs(preprocessed_data), cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Reconstructed Object (Amp)")
    plt.imshow(rec_amp, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Forward Projection of Recon")
    plt.imshow(sim_amp, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("reconstruction_results.png")
    print("Saved visualization to reconstruction_results.png")
    
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")