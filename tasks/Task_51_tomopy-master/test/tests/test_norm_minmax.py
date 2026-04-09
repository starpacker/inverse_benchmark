import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_norm_minmax import norm_minmax
from verification_utils import recursive_check

def test_norm_minmax():
    """
    Test script for norm_minmax.
    Logic:
    1. Identify data files (Outer vs Inner).
    2. Execute norm_minmax with 'outer' args.
    3. If the result is callable (closure pattern) and 'inner' data exists, execute that callable.
    4. Else, compare the immediate result with the expected output.
    """
    data_paths = ['/data/yjh/tomopy-master_sandbox/run_code/std_data/standard_data_norm_minmax.pkl']

    # 1. Identify Data Files
    outer_path = None
    inner_path = None

    for p in data_paths:
        if 'standard_data_norm_minmax.pkl' in p:
            outer_path = p
        elif 'standard_data_parent_function_norm_minmax' in p:
            inner_path = p

    if not outer_path:
        print("Error: Standard data file 'standard_data_norm_minmax.pkl' not found.")
        sys.exit(1)

    # 2. Load Outer Data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer data: {e}")
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_outer_output = outer_data.get('output')

    print(f"Running norm_minmax with args: {len(outer_args)} items, kwargs: {list(outer_kwargs.keys())}")

    # 3. Execute Outer Function
    try:
        actual_result = norm_minmax(*outer_args, **outer_kwargs)
    except Exception as e:
        print("Error executing norm_minmax:")
        traceback.print_exc()
        sys.exit(1)

    # 4. Determine Comparison Strategy
    # Scenario A: The function returns data directly (simple function).
    # Scenario B: The function returns a callable (factory/closure), requiring inner data to test fully.
    
    if callable(actual_result) and not isinstance(actual_result, (torch.Tensor, np.ndarray)):
        # This looks like a factory pattern (Scenario B)
        if not inner_path:
            print("Warning: norm_minmax returned a callable, but no inner data file (parent_function) was found to test it.")
            # We can only check if the returned object matches the expected outer output (likely the function object itself or None depending on capture)
            # However, usually data capture for factories captures the *function* as output.
            pass 
        else:
            # Load Inner Data
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"Error loading inner data: {e}")
                sys.exit(1)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_inner_output = inner_data.get('output')

            print(f"Executing resulting closure with inner args: {len(inner_args)} items")
            try:
                # Execute the closure
                final_result = actual_result(*inner_args, **inner_kwargs)
                
                # Compare closure result
                is_correct, msg = recursive_check(expected_inner_output, final_result)
                if not is_correct:
                    print(f"TEST FAILED (Closure Execution): {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED (Closure Execution)")
                    sys.exit(0)
            except Exception as e:
                print("Error executing the result closure:")
                traceback.print_exc()
                sys.exit(1)

    # Fallback / Scenario A: Compare immediate results
    is_correct, msg = recursive_check(expected_outer_output, actual_result)
    
    if not is_correct:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    test_norm_minmax()