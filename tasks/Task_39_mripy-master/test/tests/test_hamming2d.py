import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the current directory to sys.path to ensure local imports work
sys.path.append(os.getcwd())

from agent_hamming2d import hamming2d
from verification_utils import recursive_check

def test_hamming2d():
    # 1. Configuration
    data_paths = ['/data/yjh/mripy-master_sandbox/run_code/std_data/standard_data_hamming2d.pkl']
    
    # 2. Logic to distinguish between simple function vs factory pattern
    # The provided data path suggests a simple function call pattern or the start of a factory.
    # We check if there are associated 'inner' files (parent_function pattern).
    outer_data_path = None
    inner_data_path = None

    for path in data_paths:
        if 'standard_data_hamming2d.pkl' in path:
            outer_data_path = path
        elif 'standard_data_parent_function_hamming2d' in path:
            inner_data_path = path

    if not outer_data_path:
        print("Error: standard_data_hamming2d.pkl not found in data_paths.")
        sys.exit(1)

    try:
        # 3. Load Outer Data (Arguments for the main function)
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_outer_output = outer_data.get('output')

        # 4. Execute the Main Function
        print(f"Executing hamming2d with args: {outer_args}, kwargs: {outer_kwargs}")
        actual_result = hamming2d(*outer_args, **outer_kwargs)

        # 5. Determine Verification Strategy
        # If there is inner data, it implies hamming2d returned a callable (Closure/Factory).
        # If no inner data, we verify the result of hamming2d directly.
        
        if inner_data_path:
            # --- Scenario B: Factory Pattern ---
            print("Detected Factory/Closure pattern. Executing inner function...")
            
            if not callable(actual_result):
                print(f"Error: Expected hamming2d to return a callable, but got {type(actual_result)}")
                sys.exit(1)

            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_final_output = inner_data.get('output')

            print(f"Executing inner callable with args: {inner_args}, kwargs: {inner_kwargs}")
            actual_final_result = actual_result(*inner_args, **inner_kwargs)
            
            # Verify final result
            is_correct, fail_msg = recursive_check(expected_final_output, actual_final_result)
        
        else:
            # --- Scenario A: Simple Function ---
            print("Detected Simple Function pattern. Verifying direct output...")
            is_correct, fail_msg = recursive_check(expected_outer_output, actual_result)

        # 6. Final Result
        if is_correct:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {fail_msg}")
            sys.exit(1)

    except Exception as e:
        print(f"An error occurred during test execution:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_hamming2d()