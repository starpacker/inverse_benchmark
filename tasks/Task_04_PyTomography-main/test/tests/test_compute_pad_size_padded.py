import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the directory containing the target function to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the target function
from agent_compute_pad_size_padded import compute_pad_size_padded
from verification_utils import recursive_check

# Defined Data Paths
data_paths = ['/data/yjh/PyTomography-main_sandbox/run_code/std_data/standard_data_compute_pad_size_padded.pkl']

def test_compute_pad_size_padded():
    print("----------------------------------------------------------------")
    print("Starting Test: compute_pad_size_padded")
    print("----------------------------------------------------------------")

    outer_file_path = None
    inner_file_path = None

    # Classify data files
    for path in data_paths:
        if 'standard_data_compute_pad_size_padded.pkl' in path:
            outer_file_path = path
        elif 'standard_data_parent_function_compute_pad_size_padded' in path:
            inner_file_path = path

    if not outer_file_path:
        print("Error: Main data file 'standard_data_compute_pad_size_padded.pkl' not found.")
        sys.exit(1)

    # Load Outer Data
    try:
        with open(outer_file_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from {outer_file_path}")
    except Exception as e:
        print(f"Failed to load outer data: {e}")
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_outer_output = outer_data.get('output', None)

    # Execution Phase
    try:
        # Scenario A: The function returns the final result directly
        if inner_file_path is None:
            print("Running simple function execution...")
            actual_result = compute_pad_size_padded(*outer_args, **outer_kwargs)
            expected_result = expected_outer_output
        
        # Scenario B: The function returns a closure/callable, creating a factory pattern
        else:
            print("Running factory/closure pattern...")
            # 1. Create the operator
            operator = compute_pad_size_padded(*outer_args, **outer_kwargs)
            
            if not callable(operator):
                print(f"Error: Expected a callable (closure) from outer execution, but got {type(operator)}")
                sys.exit(1)

            # 2. Load Inner Data
            try:
                with open(inner_file_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from {inner_file_path}")
            except Exception as e:
                print(f"Failed to load inner data: {e}")
                sys.exit(1)

            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_result = inner_data.get('output', None)

            # 3. Execute the operator
            actual_result = operator(*inner_args, **inner_kwargs)

    except Exception as e:
        print("Execution failed with exception:")
        traceback.print_exc()
        sys.exit(1)

    # Verification
    print("Verifying results...")
    passed, msg = recursive_check(expected_result, actual_result)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    test_compute_pad_size_padded()