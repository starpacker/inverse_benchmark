import sys
import os
import dill
import torch
import numpy as np
import traceback
from agent_partial_convolution_rfft import partial_convolution_rfft
from verification_utils import recursive_check

# Global settings for consistency
torch.manual_seed(42)

def run_test():
    """
    Test script for partial_convolution_rfft.
    
    Strategy:
    1. Identify data files.
    2. Load input arguments.
    3. Execute the function.
    4. Compare results against expected output using recursive_check.
    """
    
    # 1. Configuration and Data Discovery
    data_dir = '/data/yjh/s2ISM-main_sandbox/run_code/std_data'
    base_file_name = 'standard_data_partial_convolution_rfft.pkl'
    outer_path = os.path.join(data_dir, base_file_name)
    
    # Check for inner/closure data files (Factory pattern check)
    # The generator code suggests that if the result is callable, it saves a second file with 'parent_function'.
    # However, looking at the provided function code, it returns a Tensor directly, not a callable.
    # Therefore, this is likely Scenario A (Standard function execution).
    inner_files = [f for f in os.listdir(data_dir) if 'standard_data_parent_function_partial_convolution_rfft_' in f]
    inner_path = os.path.join(data_dir, inner_files[0]) if inner_files else None

    if not os.path.exists(outer_path):
        print(f"Error: Main data file not found at {outer_path}")
        sys.exit(1)

    try:
        # 2. Load Outer Data
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_outer_result = outer_data.get('output', None)
        
        print(f"Loaded outer data from {outer_path}")
        
        # 3. Execute Function
        # Depending on whether we found inner files, the execution path differs.
        
        if inner_path:
            # Scenario B: Factory Pattern
            print("Detected Factory Pattern execution mode.")
            
            # Step 3a: Create the operator
            try:
                operator = partial_convolution_rfft(*outer_args, **outer_kwargs)
            except Exception as e:
                print(f"Failed to create operator from outer args: {e}")
                traceback.print_exc()
                sys.exit(1)
                
            if not callable(operator):
                print("Error: Expected partial_convolution_rfft to return a callable (Factory pattern), but got valid object/tensor.")
                # Fallback: Maybe it wasn't a factory after all?
                # But if inner_path exists, it implies the generator treated it as one.
                # Let's check the inner file anyway.
            
            # Step 3b: Load Inner Data
            print(f"Loading inner data from {inner_path}")
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_final_result = inner_data.get('output')
            
            # Step 3c: Execute Operator
            try:
                actual_result = operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"Failed to execute inner operator: {e}")
                traceback.print_exc()
                sys.exit(1)
                
        else:
            # Scenario A: Standard Function
            print("Detected Standard Function execution mode.")
            
            try:
                actual_result = partial_convolution_rfft(*outer_args, **outer_kwargs)
            except Exception as e:
                print(f"Failed to execute partial_convolution_rfft: {e}")
                traceback.print_exc()
                sys.exit(1)
                
            expected_final_result = expected_outer_result

        # 4. Verification
        print("Verifying results...")
        passed, msg = recursive_check(expected_final_result, actual_result)

        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()