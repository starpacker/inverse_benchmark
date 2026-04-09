import sys
import os
import dill
import numpy as np
import traceback

# Safe import for torch in case it's missing, as M_func seems to be numpy-based
try:
    import torch
except ImportError:
    torch = None

from agent_M_func import M_func
from verification_utils import recursive_check

def run_test():
    # 1. Define Data Paths
    data_paths = ['/data/yjh/DiffuserCam-Tutorial-master_sandbox/run_code/std_data/standard_data_M_func.pkl']
    
    # 2. Identify Test Strategy (Scenario A vs B)
    # Based on the provided path, we only have standard_data_M_func.pkl.
    # We need to check if the function returns a callable (Scenario B) or a value (Scenario A).
    
    outer_path = None
    inner_path = None

    for p in data_paths:
        if 'parent_function' in p:
            inner_path = p
        elif 'standard_data_M_func.pkl' in p:
            outer_path = p

    if not outer_path:
        print("Error: standard_data_M_func.pkl not found in provided paths.")
        sys.exit(1)

    # 3. Load Outer Data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading data file {outer_path}: {e}")
        sys.exit(1)

    # 4. Execute M_func
    try:
        args = outer_data['args']
        kwargs = outer_data['kwargs']
        expected_output = outer_data['output']
        
        # Run the function
        actual_output = M_func(*args, **kwargs)
        
    except Exception as e:
        print("Error executing M_func:")
        traceback.print_exc()
        sys.exit(1)

    # 5. Handle potential Factory/Closure pattern (Scenario B)
    # If the output is a callable and we have inner data, we need to execute the result.
    if callable(actual_output) and not isinstance(actual_output, (np.ndarray, list, tuple, dict)):
        if inner_path:
            print("Detected closure pattern with inner data file.")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data['args']
                inner_kwargs = inner_data['kwargs']
                expected_inner_output = inner_data['output']

                # Execute the closure
                final_result = actual_output(*inner_args, **inner_kwargs)
                
                # Check results
                passed, msg = recursive_check(expected_inner_output, final_result)
                if passed:
                    print("TEST PASSED (Closure Execution)")
                    sys.exit(0)
                else:
                    print(f"TEST FAILED (Closure Execution): {msg}")
                    sys.exit(1)
            except Exception as e:
                print(f"Error executing inner closure: {e}")
                traceback.print_exc()
                sys.exit(1)
        else:
            # Scenario: Function returned a callable, but we have no inner data to test it with.
            # We can only check if the expected output in outer_data was also a callable (or matches structure).
            # However, usually 'output' in pickle is the RESULT of the function.
            # If the original recording captured a function object, equality check is hard.
            # But based on M_func code provided: return np.real(...), it returns an array, not a function.
            # So this block is likely not needed for this specific M_func, but good for robustness.
            print("Warning: M_func returned a callable but no inner data provided. Checking callable identity/structure if possible.")
            passed, msg = recursive_check(expected_output, actual_output)
            if passed:
                 print("TEST PASSED")
                 sys.exit(0)
            else:
                 print(f"TEST FAILED: {msg}")
                 sys.exit(1)

    # 6. Standard Scenario A: Function returns data (numpy array likely)
    else:
        passed, msg = recursive_check(expected_output, actual_output)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

if __name__ == "__main__":
    run_test()