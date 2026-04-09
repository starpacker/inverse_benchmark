import sys
import os
import dill
import numpy as np
import traceback
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. Target Function Import
# -------------------------------------------------------------------------
try:
    from agent_run_inversion import run_inversion
except ImportError:
    print("Error: Could not import 'run_inversion' from 'agent_run_inversion.py'.")
    sys.exit(1)

# -------------------------------------------------------------------------
# 2. Referee / Evaluation Logic (Injected from Reference B)
# -------------------------------------------------------------------------
def evaluate_results(image_result):
    """
    Computes statistics and saves the resulting image.
    """
    # Statistics
    mean_val = np.mean(image_result)
    std_val = np.std(image_result)
    min_val = np.min(image_result)
    max_val = np.max(image_result)
    
    print(f"Evaluation Stats -> Mean: {mean_val:.4f}, Std: {std_val:.4f}, Min: {min_val:.4f}, Max: {max_val:.4f}")
    
    # Visualization
    output_filename = "oct_reconstruction_refactored.png"
    plt.figure(figsize=(10, 5))
    plt.imshow(image_result, cmap='gray', aspect='auto')
    plt.title('Refactored OCT Structure Reconstruction')
    plt.colorbar(label='Normalized Intensity')
    plt.xlabel('A-Lines')
    plt.ylabel('Depth (Z)')
    plt.savefig(output_filename)
    print(f"Result saved to {output_filename}")
    
    return output_filename

# -------------------------------------------------------------------------
# 3. Helper Functions for Data Management
# -------------------------------------------------------------------------
def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return None
    try:
        with open(file_path, 'rb') as f:
            return dill.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def identify_execution_pattern(data_paths):
    """
    Analyzes file paths to determine if this is a direct execution 
    or a factory/closure (chained) execution.
    """
    outer_data_path = None
    inner_data_paths = []

    for path in data_paths:
        if "parent_function" in path:
            inner_data_paths.append(path)
        else:
            outer_data_path = path

    return outer_data_path, inner_data_paths

# -------------------------------------------------------------------------
# 4. Main Validation Logic
# -------------------------------------------------------------------------
def main():
    # Hardcoded input paths as per prompt instructions
    data_paths = ['/data/yjh/oct-cbort-main_sandbox/run_code/std_data/standard_data_run_inversion.pkl']
    
    outer_path, inner_paths = identify_execution_pattern(data_paths)

    if not outer_path:
        print("Error: No primary outer data file found.")
        sys.exit(1)

    print(f"Loading primary data from: {outer_path}")
    outer_data = load_data(outer_path)
    if outer_data is None:
        sys.exit(1)

    try:
        # --- Execution Phase ---
        print("Executing 'run_inversion' with loaded arguments...")
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        
        # 1. Run the Agent Function
        agent_result = run_inversion(*args, **kwargs)
        
        # 2. Determine Ground Truth
        # If there are inner paths (chained execution), we would process them here.
        # Based on the provided target function code, 'run_inversion' returns an array directly,
        # so this is likely Pattern 1 (Direct Execution).
        
        if inner_paths:
            # Pattern 2: Chained Execution (Factory/Closure)
            print(f"Detected chained execution pattern with {len(inner_paths)} inner calls.")
            # For brevity in this specific task where run_inversion is a direct calculator,
            # we will assume standard direct execution unless the return is callable.
            if callable(agent_result):
                 # If we were handling closures, we would iterate inner_paths here.
                 # Given the provided source code, this branch is unlikely.
                 print("Warning: Function returned a callable, but logic assumes direct array return.")
                 pass

        # Standard Direct Execution Comparison
        std_result = outer_data.get('output')
        
        # --- Evaluation Phase ---
        print("\n--- Evaluating Agent Result ---")
        evaluate_results(agent_result)
        
        # Calculate a simple numeric metric for QA pass/fail decision (MSE)
        # Since evaluate_results returns a filename (string), we calculate a numeric score manually for validation.
        # We ensure the agent result is structurally similar to the ground truth.
        
        if std_result is not None:
            mse = np.mean((agent_result - std_result) ** 2)
            max_val = np.max(std_result)
            # Peak Signal-to-Noise Ratio (PSNR) calculation handling divide by zero
            if mse == 0:
                psnr = 100.0 # Perfect match
            else:
                psnr = 20 * np.log10(max_val / np.sqrt(mse))
                
            print(f"\nQA Metric -> MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")
            
            # Thresholds: PSNR > 30dB is usually excellent for images. 
            # Since this is a reconstruction algorithm, we expect high fidelity.
            if psnr < 30.0: 
                print("FAILURE: Agent result deviates significantly from standard result.")
                sys.exit(1)
            else:
                print("SUCCESS: Agent result matches standard result within acceptable tolerance.")
        else:
            print("Warning: No standard output found in data file for comparison. Assuming Success based on execution.")

    except Exception:
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()