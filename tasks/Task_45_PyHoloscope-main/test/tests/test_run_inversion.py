import sys
import os
import dill
import numpy as np
import math
import scipy.signal
import traceback
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# --- Target Import ---
try:
    from agent_run_inversion import run_inversion
except ImportError:
    # If not in path, try adding current directory
    sys.path.append(os.getcwd())
    from agent_run_inversion import run_inversion

# --- Reference B: The Referee (Evaluation Logic) ---
# Copied verbatim as instructed to ensure self-contained validation logic

def forward_operator_ref(field, wavelength, pixel_size, propagation_distance, precision="single"):
    """
    Propagates a complex field by a specific distance using the Angular Spectrum Method.
    This acts as the forward model: Object Plane -> Hologram Plane.
    (Reference implementation for evaluation purposes)
    """
    if precision == "double":
        dtype_c = np.complex128
    else:
        dtype_c = np.complex64

    # Ensure input is complex
    field = field.astype(dtype_c)
    
    grid_height, grid_width = field.shape
    
    # 1. Coordinate Grids for Transfer Function
    # Frequency coordinates (fx, fy)
    fx = scipy.fft.fftfreq(grid_width, d=pixel_size)
    fy = scipy.fft.fftfreq(grid_height, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy)
    
    # 2. Angular Spectrum Transfer Function (H)
    squared_sum = FX**2 + FY**2
    inv_lambda_sq = 1.0 / (wavelength**2)
    
    argument = inv_lambda_sq - squared_sum
    
    # Mask for evanescent waves
    mask = argument >= 0
    root_val = np.sqrt(np.maximum(argument, 0))
    
    # Phase factor
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
    simulated_holo_field = forward_operator_ref(
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

# --- Helper Utilities ---

def load_pkl(path):
    with open(path, 'rb') as f:
        return dill.load(f)

def run_evaluation(data_paths):
    """
    Main execution logic to test run_inversion.
    """
    
    # 1. Identify File Patterns
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        if "standard_data_run_inversion.pkl" in p and "parent_function" not in p:
            outer_path = p
        elif "standard_data_parent_function_run_inversion" in p:
            inner_paths.append(p)
            
    if not outer_path:
        print("Error: Primary data file 'standard_data_run_inversion.pkl' not found.")
        sys.exit(1)

    print(f"Loading primary data from: {outer_path}")
    outer_data = load_pkl(outer_path)
    
    # Extract arguments from the pickle
    # args: (measurements, wavelength, pixel_size, depth)
    # kwargs: precision='single', etc.
    args = outer_data.get('args', [])
    kwargs = outer_data.get('kwargs', {})
    
    # For evaluation context, we need to extract specific parameters:
    # Based on function signature: run_inversion(measurements, wavelength, pixel_size, depth, precision="single")
    # We map them positionally if kwargs are empty, or from kwargs if present.
    
    measurements = args[0] if len(args) > 0 else kwargs.get('measurements')
    wavelength = args[1] if len(args) > 1 else kwargs.get('wavelength')
    pixel_size = args[2] if len(args) > 2 else kwargs.get('pixel_size')
    depth = args[3] if len(args) > 3 else kwargs.get('depth')
    
    # 2. Execution Phase
    try:
        if not inner_paths:
            # Pattern 1: Direct Execution
            print("Executing Pattern 1: Direct Function Call")
            
            # Run Agent
            agent_result = run_inversion(*args, **kwargs)
            
            # Get Ground Truth Result
            std_result = outer_data['output']
            
        else:
            # Pattern 2: Chained Execution (Factory/Closure)
            print("Executing Pattern 2: Chained/Factory Execution")
            
            # Step A: Run Outer to get operator
            agent_operator = run_inversion(*args, **kwargs)
            
            # We assume for simplicity we take the first inner path if multiple exist
            # typically one test case runs one chain.
            inner_path = inner_paths[0]
            print(f"Loading inner data from: {inner_path}")
            inner_data = load_pkl(inner_path)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            
            # Step B: Run Inner
            agent_result = agent_operator(*inner_args, **inner_kwargs)
            std_result = inner_data['output']
            
            # Update measurements context if the inner call actually provided the data
            # (In some closure patterns, data is passed in the second call)
            # However, looking at the provided function signature, 'measurements' is in the outer call.
            # We stick to the outer params unless the result dictates otherwise.

    except Exception as e:
        print(f"CRITICAL FAILURE during execution: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Evaluation Phase
    print("Calculating Metrics...")
    
    # We need to ensure we have the necessary params for evaluation.
    # If they were None (because args mapping failed), we default or fail.
    if measurements is None or wavelength is None or pixel_size is None or depth is None:
        print("Warning: Could not extract all physical parameters from input args. Attempting simple PSNR on outputs directly.")
        # Fallback: Compare Agent Result directly to Standard Result using simple array comparison if context missing
        # But per requirements, we should use evaluate_results. We will trust the pickle had them.
    
    try:
        metrics_agent, _ = evaluate_results(agent_result, measurements, wavelength, pixel_size, depth)
        metrics_std, _ = evaluate_results(std_result, measurements, wavelength, pixel_size, depth)
        
        agent_psnr = metrics_agent['psnr']
        std_psnr = metrics_std['psnr']
        
        print(f"Scores -> Agent PSNR: {agent_psnr:.4f}, Standard PSNR: {std_psnr:.4f}")
        
        # 4. Verification
        # Allow 5% degradation margin. PSNR is logarithmic, so a small drop is significant, but floating point diffs exist.
        # Since we use the exact same forward model for evaluation as the agent likely used for inversion, 
        # results should be very close.
        
        # We check relative performance.
        # If standard PSNR is very low (garbage data), we might pass if agent is also garbage.
        # But generally, we want Agent >= Standard * 0.95 (if Standard > 0).
        
        if std_psnr > 0:
            threshold = std_psnr * 0.95
        else:
            # If standard is 0 or negative (unlikely for PSNR unless log(0)), just ensure agent isn't crashing
            threshold = -100 
            
        if agent_psnr >= threshold:
            print("Validation PASSED: Agent performance is within acceptable range.")
            sys.exit(0)
        else:
            print("Validation FAILED: Agent performance degraded significantly.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during metric calculation: {e}")
        traceback.print_exc()
        # If physics-based eval fails (e.g. shapes mismatch), fallback to direct comparison
        # But stricly, we should fail if evaluation code breaks.
        sys.exit(1)

if __name__ == "__main__":
    # Hardcoded data paths based on prompt
    data_paths = ['/data/yjh/PyHoloscope-main_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    run_evaluation(data_paths)