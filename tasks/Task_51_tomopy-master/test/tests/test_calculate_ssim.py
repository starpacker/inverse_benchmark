import sys
import os
import dill
import torch
import numpy as np
import traceback
from agent_calculate_ssim import calculate_ssim
from verification_utils import recursive_check

# Force GPU use if available for consistency with data generation environment
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def test_calculate_ssim():
    """
    Unit test for calculate_ssim function using recorded standard data.
    """
    data_paths = ['/data/yjh/tomopy-master_sandbox/run_code/std_data/standard_data_calculate_ssim.pkl']
    
    # 1. Identify Data Files
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        if 'standard_data_calculate_ssim.pkl' in p:
            outer_path = p
        elif 'standard_data_parent_function_calculate_ssim' in p:
            inner_paths.append(p)

    if not outer_path:
        print("Error: Standard data file 'standard_data_calculate_ssim.pkl' not found.")
        sys.exit(1)

    # 2. Load Outer Data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading data file {outer_path}: {e}")
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_result = outer_data.get('output', None)

    print(f"Loaded data from {outer_path}")

    # 3. Execution Strategy
    # Scenario A: The function is a direct calculator (Simple Function).
    # Scenario B: The function is a factory (returns a callable).

    try:
        # Execute the function
        actual_result = calculate_ssim(*outer_args, **outer_kwargs)
        
        # Check if the result is a callable (implying Scenario B/Factory Pattern)
        # BUT we have no inner paths provided in the data_paths list given in the prompt.
        # If inner_paths existed, we would proceed to test the returned operator.
        # Since only one file exists, we treat this as a direct calculation or a factory initialization 
        # that we can only verify against the immediate output (which might be a function object or a value).
        
        if inner_paths and callable(actual_result):
            # Scenario B: Factory Pattern with available inner data
            print("Detected Factory Pattern with inner execution data.")
            operator = actual_result
            
            for inner_path in inner_paths:
                print(f"Testing inner execution: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output', None)
                
                op_result = operator(*inner_args, **inner_kwargs)
                
                passed, msg = recursive_check(inner_expected, op_result)
                if not passed:
                    print(f"Inner Comparison FAILED for {inner_path}: {msg}")
                    sys.exit(1)
            
            print("All inner executions passed.")

        else:
            # Scenario A: Direct Calculation
            # Compare the immediate result
            passed, msg = recursive_check(expected_result, actual_result)
            if not passed:
                print(f"Comparison FAILED: {msg}")
                sys.exit(1)
            else:
                print("Direct comparison PASSED.")

    except Exception:
        traceback.print_exc()
        sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    test_calculate_ssim()