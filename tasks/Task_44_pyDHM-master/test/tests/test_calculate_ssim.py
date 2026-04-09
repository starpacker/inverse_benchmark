import sys
import os
import dill
import numpy as np
import cv2
import traceback

# Add the directory containing the target function to sys.path
# Assuming agent_calculate_ssim.py is in the current directory or python path
# If necessary, adjust path relative to the script location
sys.path.append(os.path.dirname(__file__))

from agent_calculate_ssim import calculate_ssim
from verification_utils import recursive_check

def load_data(path):
    with open(path, 'rb') as f:
        return dill.load(f)

def test_calculate_ssim():
    # 1. Define Data Paths
    data_paths = ['/data/yjh/pyDHM-master_sandbox/run_code/std_data/standard_data_calculate_ssim.pkl']
    
    outer_data_path = None
    inner_data_path = None

    # Categorize paths
    for path in data_paths:
        if 'parent_function' in path:
            inner_data_path = path
        elif 'standard_data_calculate_ssim.pkl' in path:
            outer_data_path = path

    # 2. Validation Logic
    try:
        # Load the main function inputs/outputs
        if not outer_data_path:
            print("Error: standard_data_calculate_ssim.pkl not found in provided paths.")
            sys.exit(1)
            
        outer_data = load_data(outer_data_path)
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_outer_output = outer_data.get('output')

        # Execute the function
        print(f"Executing calculate_ssim with loaded arguments...")
        try:
            actual_result = calculate_ssim(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"Execution failed with error: {e}")
            traceback.print_exc()
            sys.exit(1)

        # 3. Check if this is a factory pattern (closure) or a direct calculation
        if callable(actual_result) and not isinstance(actual_result, (np.ndarray, float, int)):
            # This is Scenario B: The function returns an operator (factory pattern)
            print("Detected Factory Pattern: calculate_ssim returned a callable.")
            
            if not inner_data_path:
                print("Error: Inner data file (parent_function_calculate_ssim) required for factory pattern test but not found.")
                sys.exit(1)
            
            inner_data = load_data(inner_data_path)
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_inner_output = inner_data.get('output')
            
            print(f"Executing inner operator with loaded arguments...")
            try:
                final_result = actual_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"Inner operator execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
                
            # Verification for Factory Pattern
            is_correct, error_msg = recursive_check(expected_inner_output, final_result)
            
        else:
            # This is Scenario A: Direct calculation
            print("Detected Direct Execution: calculate_ssim returned a value.")
            final_result = actual_result
            
            # Verification for Direct Execution
            is_correct, error_msg = recursive_check(expected_outer_output, final_result)

        # 4. Final Assertion
        if is_correct:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {error_msg}")
            sys.exit(1)

    except FileNotFoundError:
        print(f"Error: Data file not found at {outer_data_path}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_calculate_ssim()