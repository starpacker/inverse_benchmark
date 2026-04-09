import sys
import os
import dill
import torch
import numpy as np
import traceback

from agent_unpad import unpad
from verification_utils import recursive_check

# Provided data paths
data_paths = ['/data/yjh/semiblindpsfdeconv-master_sandbox/run_code/std_data/standard_data_unpad.pkl']

def load_pkl(path):
    """Helper to load pickle files safely."""
    try:
        with open(path, 'rb') as f:
            return dill.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def main():
    print("Starting test_unpad.py...")

    # 1. Identify Data Files
    outer_data_path = None
    inner_data_path = None

    for path in data_paths:
        if 'parent_function' in path:
            inner_data_path = path
        elif 'standard_data_unpad.pkl' in path:
            outer_data_path = path

    if not outer_data_path:
        print("Error: standard_data_unpad.pkl not found in provided paths.")
        sys.exit(1)

    print(f"Loading Outer Data from: {outer_data_path}")
    outer_data = load_pkl(outer_data_path)
    if not outer_data:
        sys.exit(1)

    # 2. Reconstruct Operator / Execute Function
    try:
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        
        # Execute the function with captured arguments
        print("Executing unpad with loaded arguments...")
        actual_result = unpad(*outer_args, **outer_kwargs)

    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Determine Expected Result and Compare
    # Scenario A vs B check logic
    if inner_data_path:
        # Scenario B: The first call returned a callable (closure), which we now need to execute
        print(f"Scenario B detected. Loading Inner Data from: {inner_data_path}")
        inner_data = load_pkl(inner_data_path)
        if not inner_data:
            sys.exit(1)
        
        if not callable(actual_result):
            print(f"Error: Expected unpad to return a callable for inner execution, but got {type(actual_result)}")
            sys.exit(1)
            
        try:
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output')
            
            print("Executing returned operator with inner arguments...")
            final_result = actual_result(*inner_args, **inner_kwargs)
            
            # Verification
            is_match, msg = recursive_check(expected_output, final_result)
            
        except Exception as e:
            print(f"Inner execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
            
    else:
        # Scenario A: The first call returned the final result
        print("Scenario A detected (Direct execution).")
        expected_output = outer_data.get('output')
        final_result = actual_result
        
        # Verification
        is_match, msg = recursive_check(expected_output, final_result)

    # 4. Final Result
    if is_match:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()