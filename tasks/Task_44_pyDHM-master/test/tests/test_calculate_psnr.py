import sys
import os
import dill
import torch
import numpy as np
import traceback
from agent_calculate_psnr import calculate_psnr
from verification_utils import recursive_check

# List of data files to analyze
data_paths = ['/data/yjh/pyDHM-master_sandbox/run_code/std_data/standard_data_calculate_psnr.pkl']

def load_data(path):
    with open(path, 'rb') as f:
        return dill.load(f)

def run_test():
    # 1. Identify Data Files
    # The provided list contains the 'outer' or 'direct' call data.
    # We look for a file ending exactly with 'standard_data_calculate_psnr.pkl'
    outer_data_path = None
    inner_data_path = None

    for p in data_paths:
        if p.endswith('standard_data_calculate_psnr.pkl'):
            outer_data_path = p
        elif 'standard_data_parent_function_calculate_psnr' in p:
            inner_data_path = p

    if not outer_data_path:
        print("Error: standard_data_calculate_psnr.pkl not found in data_paths.")
        sys.exit(1)

    # 2. Load Outer Data
    try:
        outer_data = load_data(outer_data_path)
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        outer_expected = outer_data.get('output', None)
    except Exception as e:
        print(f"Error loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Execute Target Function
    try:
        # Based on the provided code, calculate_psnr is a direct function that returns a value (PSNR score),
        # not a factory returning a function.
        # So we run it and compare directly against outer_expected.
        
        # However, to be robust to the 'Factory Pattern' mentioned in instructions:
        # If inner_data_path exists, it implies calculate_psnr returned a callable (Scenario B).
        # If not, it returned the result directly (Scenario A).
        
        actual_result_or_op = calculate_psnr(*outer_args, **outer_kwargs)

        if inner_data_path:
            # Scenario B: Factory Pattern
            if not callable(actual_result_or_op):
                print(f"Error: Expected a callable return from calculate_psnr (based on existence of inner data), but got {type(actual_result_or_op)}")
                sys.exit(1)
            
            # Load inner data to execute the returned operator
            try:
                inner_data = load_data(inner_data_path)
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output', None)
            except Exception as e:
                print(f"Error loading inner data: {e}")
                sys.exit(1)
            
            # Execute inner
            actual_result = actual_result_or_op(*inner_args, **inner_kwargs)
            expected_result = inner_expected
            
        else:
            # Scenario A: Simple Function (This matches the provided calculate_psnr source code)
            actual_result = actual_result_or_op
            expected_result = outer_expected

    except Exception as e:
        print(f"Error executing function: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verify Results
    try:
        passed, msg = recursive_check(expected_result, actual_result)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            # Debug info
            print(f"Expected type: {type(expected_result)}")
            print(f"Actual type: {type(actual_result)}")
            if isinstance(expected_result, (float, np.floating)):
                print(f"Expected val: {expected_result}")
                print(f"Actual val: {actual_result}")
            sys.exit(1)
    except Exception as e:
        print(f"Error during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()