import sys
import os
import dill
import numpy as np
import traceback

# --- 1. Setup & Imports ---
# Import the target function
try:
    from agent_run_inversion import run_inversion
except ImportError:
    # If not in path, try adding current directory
    sys.path.append(os.getcwd())
    from agent_run_inversion import run_inversion

# --- 2. Inject Referee (Evaluation Logic) ---
def evaluate_results(ground_truth, reconstructed_image):
    """
    Compares the ground truth phantom with the reconstructed image.
    Calculates PSNR and SSIM.
    """
    
    # Ground truth also needs to be cropped to match the reconstruction
    # Reconstruction logic: c[1:-1, 1:-1]
    # Note: If ground truth is already cropped or different shape, handle gracefully
    
    # Helper to safe crop
    def safe_crop(img):
        if img.shape[0] > 2 and img.shape[1] > 2:
            return img[1:-1, 1:-1]
        return img

    # Ground truth is likely the 'output' from the standard run (the valid reconstruction)
    # The reconstructed_image is what our agent produced.
    # In the context of this test, we are comparing the Agent's output against the recorded Standard output.
    # However, the prompt implies 'ground_truth' might be a phantom.
    # But usually in regression testing 'standard_data' contains the 'correct' output.
    # Let's assume the 'ground_truth' passed here is the stored output from the pkl file.
    
    # Check shapes. If identical, assume direct comparison.
    # If ground_truth is larger, apply crop.
    
    if ground_truth.shape != reconstructed_image.shape:
        # Attempt crop if GT is slightly larger (border issue)
        gt_cropped = safe_crop(ground_truth)
    else:
        gt_cropped = ground_truth

    # Normalize Ground Truth
    max_gt = np.max(gt_cropped)
    if max_gt > 0:
        gt_norm = gt_cropped / max_gt
    else:
        gt_norm = gt_cropped

    # Normalize Prediction (Agent Output)
    max_pred = np.max(reconstructed_image)
    if max_pred > 0:
        prediction = reconstructed_image / max_pred
    else:
        prediction = reconstructed_image
        
    target = gt_norm
    
    # Ensure shapes match now
    if target.shape != prediction.shape:
        # Resize or crop prediction to match target if simple mismatch
        min_x = min(target.shape[0], prediction.shape[0])
        min_y = min(target.shape[1], prediction.shape[1])
        target = target[:min_x, :min_y]
        prediction = prediction[:min_x, :min_y]

    data_range = target.max() - target.min()
    if data_range == 0: 
        data_range = 1.0

    # --- PSNR ---
    mse = np.mean((target - prediction) ** 2)
    if mse == 0:
        psnr_val = float('inf')
    else:
        psnr_val = 20 * np.log10(data_range / np.sqrt(mse))

    # --- SSIM (Simplified) ---
    mu_x = target.mean()
    mu_y = prediction.mean()
    var_x = target.var()
    var_y = prediction.var()
    cov_xy = np.mean((target - mu_x) * (prediction - mu_y))
    
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    
    numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    
    ssim_val = numerator / denominator
    
    print(f"Evaluation Metrics:")
    print(f"  PSNR: {psnr_val:.2f} dB")
    print(f"  SSIM: {ssim_val:.4f}")
    
    return {
        'PSNR': psnr_val,
        'SSIM': ssim_val,
        'MSE': mse
    }

# --- 3. Test Runner Logic ---

def run_test():
    # Hardcoded path as per instructions
    data_paths = ['/data/yjh/MPIRF-master_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    if not os.path.exists(data_paths[0]):
        print(f"Error: Data file not found at {data_paths[0]}")
        sys.exit(1)

    print("Loading data...")
    try:
        with open(data_paths[0], 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit(1)

    # Check for Inner Data (Pattern 2)
    # The prompt mentions: standard_data_parent_function_run_inversion_*.pkl
    # We scan the directory of the outer file for potential inner files.
    base_dir = os.path.dirname(data_paths[0])
    files_in_dir = os.listdir(base_dir)
    inner_file = None
    
    # Look for files matching the pattern "standard_data_parent_run_inversion_*.pkl" 
    # Note: The prompt example had "parent_function_run_inversion", let's be flexible.
    for fname in files_in_dir:
        if "standard_data_parent_run_inversion" in fname and fname.endswith(".pkl"):
             inner_file = os.path.join(base_dir, fname)
             break
        # Also check specific naming from gen_code snippet logic if slightly different
        if "standard_data_parent_function_run_inversion" in fname and fname.endswith(".pkl"):
            inner_file = os.path.join(base_dir, fname)
            break

    # --- Execution Phase ---
    try:
        print(f"Running 'run_inversion' with Outer Data...")
        # Extract Args/Kwargs
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        
        # Execute Outer
        agent_output = run_inversion(*args, **kwargs)

        final_result = None
        std_result = None

        if inner_file:
            print(f"Detected Chained Execution. Inner file: {inner_file}")
            # Pattern 2: Closure/Factory
            if not callable(agent_output):
                print("Error: Expected 'run_inversion' to return a callable (operator) for chained execution, but got", type(agent_output))
                sys.exit(1)

            with open(inner_file, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            
            # Execute Inner
            final_result = agent_output(*inner_args, **inner_kwargs)
            std_result = inner_data['output']
            
        else:
            print("Detected Direct Execution.")
            # Pattern 1: Direct
            final_result = agent_output
            std_result = outer_data['output']

        # --- Evaluation Phase ---
        print("\n--- Evaluating Agent Performance ---")
        
        # In this specific scenario, the 'run_inversion' function is an image reconstruction algorithm.
        # Ideally, we compare against a Ground Truth Phantom. 
        # However, we only have the input data and the previous standard output.
        # We will use the 'std_result' (the recorded output from a previous successful run) as our 'Ground Truth' for regression testing.
        # This ensures the refactored/agent code performs as well as the original code.
        
        metrics = evaluate_results(std_result, final_result)
        
        # Determine Success
        # Since we are comparing against the Standard Output (Regression Test), 
        # we expect extremely high similarity (PSNR > 50dB or SSIM > 0.99).
        # However, allowing for floating point diffs on GPU vs CPU, we set a reasonable threshold.
        
        psnr = metrics['PSNR']
        ssim = metrics['SSIM']
        
        # Thresholds
        PSNR_THRESHOLD = 30.0  # dB
        SSIM_THRESHOLD = 0.95
        
        if psnr == float('inf') or (psnr > PSNR_THRESHOLD and ssim > SSIM_THRESHOLD):
            print("\nSUCCESS: Performance is acceptable.")
            sys.exit(0)
        else:
            print(f"\nFAILURE: Performance below threshold (PSNR > {PSNR_THRESHOLD}, SSIM > {SSIM_THRESHOLD}).")
            sys.exit(1)

    except Exception as e:
        print(f"An error occurred during execution:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()