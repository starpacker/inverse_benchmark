import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the directory containing the target function to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_compute_pad_size import compute_pad_size
from verification_utils import recursive_check

# List of data paths provided
data_paths = ['/data/yjh/PyTomography-main_sandbox/run_code/std_data/standard_data_compute_pad_size.pkl']

def test_compute_pad_size():
    print("----------------------------------------------------------------")
    print("Starting Unit Test: test_compute_pad_size")
    print("----------------------------------------------------------------")

    # 1. Identify Data Files
    outer_path = None
    inner_paths = []

    for p in data_paths:
        if 'standard_data_compute_pad_size.pkl' in p:
            outer_path = p
        elif 'standard_data_parent_function_compute_pad_size' in p:
            inner_paths.append(p)

    if not outer_path:
        print("Error: Standard data file 'standard_data_compute_pad_size.pkl' not found.")
        sys.exit(1)

    # 2. Load Outer Data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"Error loading outer data: {e}")
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')

    # 3. Execution Strategy
    # Scenario A: Simple Function (Direct execution)
    # Scenario B: Factory/Closure (Execution returns a callable, which we then test with inner data)
    
    try:
        print("Executing compute_pad_size with outer arguments...")
        actual_result = compute_pad_size(*outer_args, **outer_kwargs)
        
        # Check if the result is a callable (indicating a factory pattern) and we have inner data to test it
        if callable(actual_result) and inner_paths:
            print("Detected Factory Pattern. Result is callable. Testing inner execution paths...")
            
            for inner_path in inner_paths:
                print(f"  -> Testing inner data: {inner_path}")
                try:
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                    
                    inner_args = inner_data.get('args', [])
                    inner_kwargs = inner_data.get('kwargs', {})
                    inner_expected = inner_data.get('output')
                    
                    inner_actual = actual_result(*inner_args, **inner_kwargs)
                    
                    passed, msg = recursive_check(inner_expected, inner_actual)
                    if not passed:
                        print(f"    FAILED: {msg}")
                        sys.exit(1)
                    else:
                        print("    PASSED inner check.")
                        
                except Exception as inner_e:
                    print(f"    Error during inner execution: {inner_e}")
                    traceback.print_exc()
                    sys.exit(1)
            
            print("All inner tests passed.")
            
        else:
            # Scenario A: Simple Direct Return
            print("Standard execution mode (non-factory or no inner data found).")
            passed, msg = recursive_check(expected_output, actual_result)
            if not passed:
                print(f"FAILED: {msg}")
                print(f"Expected: {expected_output}")
                print(f"Actual:   {actual_result}")
                sys.exit(1)
            else:
                print("Verification Successful.")

    except Exception as e:
        print(f"An error occurred during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("----------------------------------------------------------------")
    print("TEST PASSED")
    print("----------------------------------------------------------------")
    sys.exit(0)

if __name__ == "__main__":
    test_compute_pad_size()