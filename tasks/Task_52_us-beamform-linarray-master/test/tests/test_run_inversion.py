import sys
import os
import dill
import numpy as np
import traceback
import logging

# --- TARGET IMPORT ---
from agent_run_inversion import run_inversion

# --- DEPENDENCIES FOR EVALUATION ---
import matplotlib.pyplot as plt

# --- INJECTED REFEREE (Evaluation Function) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_results(result_img, reference_img):
    """
    Calculates PSNR and RMSE, and saves a comparison plot.
    """
    # Ensure inputs are numpy arrays
    result_img = np.array(result_img)
    reference_img = np.array(reference_img)

    # Handle Tuple Returns: if the function returns (image, x, z), extract image (index 0)
    if result_img.ndim == 1 and len(result_img.shape) > 0 and isinstance(result_img[0], (np.ndarray, list)):
        # Heuristic: usually the first element is the main result
        result_img = np.array(result_img[0])
    
    if reference_img.ndim == 1 and len(reference_img.shape) > 0 and isinstance(reference_img[0], (np.ndarray, list)):
        reference_img = np.array(reference_img[0])

    # Resize if shapes don't match (though they should for valid comparisons)
    if result_img.shape != reference_img.shape:
        print(f"Warning: Shapes differ. Result: {result_img.shape}, Ref: {reference_img.shape}. Attempting to compare intersection or flatten.")
        min_r = min(result_img.shape[0], reference_img.shape[0])
        min_c = min(result_img.shape[1], reference_img.shape[1])
        result_img = result_img[:min_r, :min_c]
        reference_img = reference_img[:min_r, :min_c]

    # Calculate Metrics
    mse = np.mean((result_img - reference_img) ** 2)
    if mse == 0:
        psnr = float('inf')
        rmse = 0.0
    else:
        data_range = 255.0 # Since images are uint8
        psnr = 20 * np.log10(data_range / np.sqrt(mse))
        rmse = np.sqrt(mse)
    
    print(f"\nEvaluation Results:")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"RMSE: {rmse:.2f}")
    
    return psnr, rmse

# --- MAIN TEST SCRIPT ---
def main():
    data_paths = ['/data/yjh/us-beamform-linarray-master_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    # 1. Identify Data Pattern
    outer_data_path = None
    inner_data_path = None
    
    for path in data_paths:
        if 'parent_function' in path:
            inner_data_path = path
        else:
            outer_data_path = path
            
    if not outer_data_path:
        print("Error: No primary data file found.")
        sys.exit(1)

    try:
        # 2. Load Outer Data
        print(f"Loading Outer Data: {outer_data_path}")
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
            
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        std_outer_output = outer_data.get('output', None)

        # 3. Execution Logic
        if inner_data_path:
            # Chained Execution Pattern
            print(f"Detected Chained Execution. Inner Data: {inner_data_path}")
            
            # Run Step 1: Get Operator
            print("Running Step 1 (Outer)...")
            operator = run_inversion(*outer_args, **outer_kwargs)
            
            # Load Inner Data
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            std_final_result = inner_data.get('output', None)
            
            # Run Step 2: Get Final Result
            print("Running Step 2 (Inner)...")
            final_result = operator(*inner_args, **inner_kwargs)
            
        else:
            # Direct Execution Pattern
            print("Detected Direct Execution.")
            
            # Run Function
            print("Running Target Function...")
            final_result = run_inversion(*outer_args, **outer_kwargs)
            std_final_result = std_outer_output

        # 4. Evaluation
        # If the result is a tuple (image, x, z), we need to extract the image part for metric calculation
        # The evaluate_results function handles shape extraction, but let's be explicit if we can.
        # Based on target code: return image_final, x_sc, z_sc
        
        agent_img = final_result
        std_img = std_final_result

        # Check for tuple unpacking
        if isinstance(final_result, tuple):
            agent_img = final_result[0]
        if isinstance(std_final_result, tuple):
            std_img = std_final_result[0]

        print("\n--- Evaluating Agent Performance ---")
        psnr_agent, rmse_agent = evaluate_results(agent_img, std_img)
        
        # Since we are comparing against the Standard Output (Ground Truth), 
        # a perfect match means PSNR -> Infinity, RMSE -> 0.
        # This validates that the Agent logic reproduces the Standard logic.
        
        # However, the prompt implies "validate performance integrity" usually against a Ground Truth Dataset.
        # But here 'std_result' IS the Ground Truth from the pickle.
        # So we are essentially checking regression.
        
        print(f"Scores -> PSNR: {psnr_agent}, RMSE: {rmse_agent}")

        # 5. Success Criteria
        # Since we are validating code integrity (regression test), we expect very high similarity.
        # PSNR > 40dB is typically excellent/indistinguishable.
        # PSNR > 100dB implies near-perfect digital identity.
        
        # If the code has changed optimization slightly (e.g. float precision), PSNR might not be infinite but should be high.
        # If the code logic is broken, PSNR will be low.
        
        if psnr_agent < 30.0: # Arbitrary threshold for "acceptable regression"
            print("FAILED: PSNR is too low (< 30 dB). Output deviates significantly from standard.")
            sys.exit(1)
        
        print("SUCCESS: Performance check passed.")
        sys.exit(0)

    except Exception as e:
        print(f"Execution Failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()