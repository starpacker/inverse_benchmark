import sys
import os
import dill
import numpy as np
import cv2
import traceback
from math import pi, sqrt, log10

# Add the current directory to sys.path to ensure local imports work
sys.path.append(os.getcwd())

# Import the target function
try:
    from agent_run_inversion import run_inversion
except ImportError:
    print("Error: Could not import 'run_inversion' from 'agent_run_inversion.py'")
    sys.exit(1)

# --- Referee Function Injection (Reference B) ---

def calculate_psnr(img1, img2):
    """Calculates PSNR between two normalized images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0 
    return 20 * log10(PIXEL_MAX / sqrt(mse))

def calculate_ssim(img1, img2):
    """
    Calculates SSIM between two images.
    """
    C1 = (0.01 * 1)**2
    C2 = (0.03 * 1)**2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Gaussian kernel for local mean
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def evaluate_results(reconstructed_field, gt_amp, gt_phase):
    """
    Calculates metrics and saves images.
    
    Returns:
        metrics (dict): Dictionary containing PSNR and SSIM.
    """
    reconstructed_amplitude = np.abs(reconstructed_field)
    reconstructed_phase = np.angle(reconstructed_field)
    
    # Clip result to valid range [0, 1] for amplitude comparison
    reconstructed_amplitude_clipped = np.clip(reconstructed_amplitude, 0, 1)
    
    psnr_val = calculate_psnr(gt_amp, reconstructed_amplitude_clipped)
    ssim_val = calculate_ssim(gt_amp, reconstructed_amplitude_clipped)
    
    print(f"Amplitude PSNR: {psnr_val:.2f} dB")
    print(f"Amplitude SSIM: {ssim_val:.4f}")
    
    # Save outputs
    cv2.imwrite('output_gt_amp.png', (gt_amp * 255).astype(np.uint8))
    cv2.imwrite('output_reconstruction_amp.png', (reconstructed_amplitude_clipped * 255).astype(np.uint8))
    
    # Visualize Phase (Normalize to 0-255 for visualization)
    gt_phase_norm = ((gt_phase - gt_phase.min()) / (gt_phase.max() - gt_phase.min()) * 255).astype(np.uint8)
    rec_phase_norm = ((reconstructed_phase - reconstructed_phase.min()) / (reconstructed_phase.max() - reconstructed_phase.min()) * 255).astype(np.uint8)
    
    cv2.imwrite('output_gt_phase.png', gt_phase_norm)
    cv2.imwrite('output_reconstruction_phase.png', rec_phase_norm)
    
    print("Saved output images: output_gt_amp.png, output_reconstruction_amp.png, output_gt_phase.png, output_reconstruction_phase.png")
    
    return {"PSNR": psnr_val, "SSIM": ssim_val}

# --- Helper Functions for Data Handling ---

def recursive_detach(obj):
    """Detaches tensors from GPU/Computation graph if they are torch tensors."""
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy()
    except ImportError:
        pass
    
    if isinstance(obj, list):
        return [recursive_detach(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(recursive_detach(x) for x in obj)
    if isinstance(obj, dict):
        return {k: recursive_detach(v) for k, v in obj.items()}
    return obj

def load_pkl(path):
    with open(path, 'rb') as f:
        return dill.load(f)

# --- Main Test Logic ---

def main():
    # Defined data paths
    data_paths = ['/data/yjh/pyDHM-master_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # Identify Outer and Inner data
    outer_data_path = None
    inner_data_path = None

    for path in data_paths:
        if 'parent_function_run_inversion' in path:
            inner_data_path = path
        elif 'standard_data_run_inversion.pkl' in path:
            outer_data_path = path

    if not outer_data_path:
        print("Error: Primary data file 'standard_data_run_inversion.pkl' not found.")
        sys.exit(1)

    print(f"Loading Outer Data: {outer_data_path}")
    try:
        outer_data = load_pkl(outer_data_path)
    except Exception as e:
        print(f"Failed to load outer data: {e}")
        sys.exit(1)

    # Prepare inputs for the agent function
    args = outer_data.get('args', [])
    kwargs = outer_data.get('kwargs', {})
    
    # Detach any torch tensors to numpy if necessary
    args = recursive_detach(args)
    kwargs = recursive_detach(kwargs)

    print("Executing 'run_inversion' with Outer Data...")
    try:
        # Run the agent's code
        agent_result = run_inversion(*args, **kwargs)
    except Exception as e:
        print(f"Execution of 'run_inversion' failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    final_result = agent_result
    std_result = outer_data.get('output')

    # Handle Chained Execution if Inner Data exists
    if inner_data_path:
        print(f"Inner Data detected: {inner_data_path}")
        if not callable(agent_result):
            print("Error: 'run_inversion' was expected to return a callable for chained execution, but it didn't.")
            sys.exit(1)
        
        try:
            inner_data = load_pkl(inner_data_path)
        except Exception as e:
            print(f"Failed to load inner data: {e}")
            sys.exit(1)
            
        inner_args = inner_data.get('args', [])
        inner_kwargs = inner_data.get('kwargs', {})
        inner_args = recursive_detach(inner_args)
        inner_kwargs = recursive_detach(inner_kwargs)
        
        print("Executing Resulting Operator with Inner Data...")
        try:
            final_result = agent_result(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"Execution of the returned operator failed: {e}")
            traceback.print_exc()
            sys.exit(1)
            
        std_result = inner_data.get('output')

    # --- Ground Truth Handling ---
    # In DHM/Inverse problems, the input to the generation function often serves as the Ground Truth.
    # We need to extract the ground truth amplitude and phase.
    # Looking at the signature of run_inversion(I_stack, z, wavelength, dx, dy), 
    # it reconstructs the field.
    # However, to evaluate 'evaluate_results(reconstructed_field, gt_amp, gt_phase)', we need GT.
    # The 'std_result' (captured output) is the reconstructed field from the ORIGINAL valid run.
    # Ideally, we should compare the Agent's result against the Standard Result (which we trust is correct/GT).
    
    # Since we don't have explicit external "GT images" in the pickle, 
    # we treat the `std_result` (the output from the captured run) as the Ground Truth Reference 
    # for the purpose of regression testing.
    
    # We will derive gt_amp and gt_phase from the std_result (the historical successful output).
    if std_result is None:
        print("Error: Standard result (Ground Truth) is None.")
        sys.exit(1)
        
    std_result_detached = recursive_detach(std_result)
    gt_amp = np.abs(std_result_detached)
    gt_phase = np.angle(std_result_detached)
    
    # Clip GT Amp similarly for consistency
    gt_amp = np.clip(gt_amp, 0, 1)

    print("\n--- Evaluating Agent Results against Stored Standard Result (Proxy GT) ---")
    try:
        metrics = evaluate_results(final_result, gt_amp, gt_phase)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    psnr = metrics.get('PSNR', 0)
    ssim = metrics.get('SSIM', 0)

    # Define Thresholds
    # Since we are comparing against the standard output of the same code (presumably), 
    # we expect very high similarity. However, floating point differences on GPU/CPU might exist.
    PSNR_THRESHOLD = 30.0  # dB
    SSIM_THRESHOLD = 0.95

    print(f"\nFinal Metrics -> PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
    
    if psnr < PSNR_THRESHOLD or ssim < SSIM_THRESHOLD:
        print("FAILURE: Performance metrics are below acceptable thresholds.")
        sys.exit(1)
    else:
        print("SUCCESS: Performance metrics are acceptable.")
        sys.exit(0)

if __name__ == "__main__":
    main()