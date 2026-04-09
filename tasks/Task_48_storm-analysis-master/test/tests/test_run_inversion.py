import sys
import os
import dill
import numpy as np
import traceback
import scipy.optimize
import scipy.ndimage

# Add the directory containing the agent code to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the target function
try:
    from agent_run_inversion import run_inversion
except ImportError:
    print("Error: Could not import 'run_inversion' from 'agent_run_inversion.py'")
    sys.exit(1)

# --- INJECTED HELPER FUNCTIONS (From Reference C) ---
# We need to ensure symmetric_gaussian_2d is available if run_inversion depends on it internally
# or if it's needed for the test environment.
def symmetric_gaussian_2d(xy, background, height, center_x, center_y, width):
    """
    Explicit mathematical definition of a 2D Symmetric Gaussian.
    f(x,y) = background + height * exp( -2 * ( ((x-cx)/w)^2 + ((y-cy)/w)^2 ) )
    """
    x, y = xy
    g = background + height * np.exp(-2 * (((center_x - x) / width) ** 2 + ((center_y - y) / width) ** 2))
    return g.ravel()

# --- INJECTED EVALUATION LOGIC (From Reference B) ---
def evaluate_results(original, reconstructed):
    """
    Calculates PSNR.
    Here, 'original' is the raw input image (ground truth signal + noise),
    and 'reconstructed' is the image generated from the fitted parameters.
    
    However, the prompt's `evaluate_results` signature implies comparing two images.
    The `run_inversion` returns (fitted_params, bg_img).
    We need to reconstruct the image from params to compare against the original input.
    
    Wait, the Prompt says: "Use a Standard Evaluation Function (evaluate_results) to compare the Agent's quality against the Ground Truth."
    The provided `evaluate_results` takes `(original, reconstructed)`.
    
    The standard output of `run_inversion` is `(fitted_params, bg_img)`.
    The standard output of the stored pickle data is also `(fitted_params, bg_img)`.
    
    To evaluate:
    1. We need the original image (which is input arg[0]).
    2. We need to reconstruct the synthetic image from the Agent's `fitted_params`.
    3. We need to reconstruct the synthetic image from the Standard `fitted_params`.
    4. Calculate PSNR for both against the original image (or just compare the reconstructed signals).
    
    Let's adapt the evaluation workflow to:
    1. Reconstruct image from params.
    2. Run `evaluate_results(original_image, reconstructed_image)`.
    """
    # Safety check for dimensions
    if original.shape != reconstructed.shape:
        # Resize reconstructed to match original if needed, or return worst score
        return 0.0

    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    
    data_range = np.max(original) - np.min(original)
    if data_range == 0:
        return 0.0
        
    psnr = 20 * np.log10(data_range / np.sqrt(mse))
    return psnr

def reconstruct_image_from_params(image_shape, params, background_img):
    """
    Helper to turn fitted parameters back into an image for PSNR comparison.
    """
    h, w = image_shape
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    reconstructed = np.copy(background_img) # Start with estimated background
    
    for p in params:
        # p = [bg, height, cx, cy, width]
        # Note: The Gaussian function adds 'background' parameter. 
        # Since we already have a background image, we usually just add the Gaussian peak (height * exp...).
        # However, the optimization included a local 'bg' offset. 
        # To avoid double counting global vs local background, a common strategy in evaluation 
        # is just to generate the peaks and add to the smoothed background.
        # But the fitted 'bg' is the local offset.
        
        # Let's strictly use the formula:
        # symmetric_gaussian_2d returns: bg + height * exp...
        # We can add this to an empty canvas, but we need to handle overlaps carefully.
        # Simple approach: Generate the gaussian on a blank canvas, subtract the 'bg' param (to get just the peak),
        # and add to the base.
        
        bg, height, cx, cy, width = p
        # Generate peak only
        peak_values = height * np.exp(-2 * (((cx - x_grid) / width) ** 2 + ((cy - y_grid) / width) ** 2))
        reconstructed += peak_values
        
    return reconstructed

def run_test():
    # 1. Setup Data Paths
    data_paths = ['/data/yjh/storm-analysis-master_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    outer_data_path = None
    inner_data_path = None

    # Identify files
    for path in data_paths:
        if "parent_function" in path:
            inner_data_path = path
        else:
            outer_data_path = path

    if not outer_data_path:
        print("Error: No standard_data_run_inversion.pkl found.")
        sys.exit(1)

    print(f"Loading data from: {outer_data_path}")
    
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. Extract Inputs
    args = outer_data.get('args', [])
    kwargs = outer_data.get('kwargs', {})
    std_output = outer_data.get('output', None)
    
    # The first argument is usually the image
    if len(args) > 0:
        original_image = args[0]
    elif 'image' in kwargs:
        original_image = kwargs['image']
    else:
        print("Error: Could not locate input image in args/kwargs.")
        sys.exit(1)

    # 3. Execution Phase
    print("Running Agent run_inversion...")
    try:
        # Pattern 1: Direct Execution
        agent_output = run_inversion(*args, **kwargs)
        
        # If there was a closure pattern, we would handle it here, 
        # but run_inversion is a standard function returning (params, bg).
        final_result = agent_output
        std_result = std_output
        
    except Exception as e:
        print(f"Error executing run_inversion: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Evaluation Phase
    # run_inversion returns: (fitted_params, bg_img)
    if not isinstance(final_result, (tuple, list)) or len(final_result) != 2:
        print(f"Error: Expected tuple (params, bg_img), got {type(final_result)}")
        sys.exit(1)

    agent_params, agent_bg = final_result
    std_params, std_bg = std_result
    
    # Reconstruct images for comparison
    print("Reconstructing images from fitted parameters...")
    agent_reconstruction = reconstruct_image_from_params(original_image.shape, agent_params, agent_bg)
    std_reconstruction = reconstruct_image_from_params(original_image.shape, std_params, std_bg)
    
    # Evaluate PSNR against the original noisy image (or we could compare reconstructions directly)
    # Usually, we want to know how well the fit explains the data.
    score_agent = evaluate_results(original_image, agent_reconstruction)
    score_std = evaluate_results(original_image, std_reconstruction)
    
    print("-" * 40)
    print(f"Scores (PSNR) -> Agent: {score_agent:.4f}, Standard: {score_std:.4f}")
    print("-" * 40)

    # 5. Verification
    # PSNR: Higher is better.
    # We allow a small margin of error (e.g., 5-10% drop, or practically 1-2 dB difference).
    # Since floating point differences in optimization can happen across machines.
    
    # If the standard failed to find anything (score 0 or very low), and agent also is low, it's a pass.
    if score_std == 0:
        print("Standard score is 0. Assuming Pass if Agent runs without error.")
        sys.exit(0)
        
    if score_agent == float('inf'):
         print("Agent achieved perfect reconstruction (Infinite PSNR). PASS.")
         sys.exit(0)

    # Threshold: Allow 10% degradation or within 3dB
    threshold = 0.9
    
    if score_agent >= score_std * threshold:
        print("Test PASSED: Performance is within acceptable range.")
        sys.exit(0)
    else:
        print("Test FAILED: Performance degradation detected.")
        sys.exit(1)

if __name__ == "__main__":
    run_test()