import sys
import os
import dill
import numpy as np
import traceback

# Ensure the environment can find the agent code
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_estimate_cov import estimate_cov
from verification_utils import recursive_check

def run_test():
    """
    Unit test for estimate_cov function using recorded data.
    """
    data_paths = ['/data/yjh/spectral_ct_examples-master_sandbox/run_code/std_data/standard_data_estimate_cov.pkl']
    
    # Identify Data Files
    outer_path = None
    inner_path = None

    for path in data_paths:
        if 'parent_function' in path:
            inner_path = path
        elif 'standard_data_estimate_cov.pkl' in path:
            outer_path = path

    if not outer_path:
        print("TEST FAILED: Standard data file (outer) not found in provided paths.")
        sys.exit(1)

    # --- Phase 1: Load and Execute Outer Data ---
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        # Execute target function
        print(f"Executing estimate_cov with {len(outer_args)} args and {len(outer_kwargs)} kwargs...")
        actual_result = estimate_cov(*outer_args, **outer_kwargs)

    except Exception as e:
        print(f"TEST FAILED: Execution of estimate_cov failed.\n{traceback.format_exc()}")
        sys.exit(1)

    # --- Phase 2: Check for Factory Pattern (Scenario B) ---
    # The function estimate_cov returns a scalar (sigma), not a callable.
    # However, we check if inner data exists just in case the provided code was a wrapper or changed.
    
    if inner_path:
        # If inner path exists, it implies the result of outer was a callable (Factory pattern)
        if not callable(actual_result):
            print("TEST FAILED: Data implies Factory Pattern (inner file exists), but estimate_cov did not return a callable.")
            sys.exit(1)
            
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_inner_output = inner_data.get('output')

            print(f"Executing resulting operator with {len(inner_args)} args and {len(inner_kwargs)} kwargs...")
            final_result = actual_result(*inner_args, **inner_kwargs)
            
            # Update expectation for verification
            actual_result = final_result
            expected_output = expected_inner_output

        except Exception as e:
             print(f"TEST FAILED: Execution of inner operator failed.\n{traceback.format_exc()}")
             sys.exit(1)

    # --- Phase 3: Verification ---
    try:
        passed, msg = recursive_check(expected_output, actual_result)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: Verification failed.\n{msg}")
            sys.exit(1)
    except Exception as e:
        print(f"TEST FAILED: Error during verification comparison.\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()