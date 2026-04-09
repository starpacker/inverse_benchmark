import sys
import os
import dill
import numpy as np
import torch
import traceback

# Add the directory containing the agent code to python path if necessary
# Assuming execution happens in the directory where agent__get_image_quadrants.py is located
# or it is importable.

try:
    from agent__get_image_quadrants import _get_image_quadrants
    from verification_utils import recursive_check
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# Provided Data Paths
DATA_PATHS = ['/data/yjh/PyAbel-master_sandbox/run_code/std_data/standard_data__get_image_quadrants.pkl']

def run_test():
    print("Starting test for _get_image_quadrants...")
    
    # 1. FILE ANALYSIS
    outer_path = None
    inner_paths = []

    for p in DATA_PATHS:
        if 'parent_function' in p:
            inner_paths.append(p)
        elif 'standard_data__get_image_quadrants.pkl' in p:
            outer_path = p

    if outer_path is None:
        print("CRITICAL ERROR: Main data file (outer) not found in provided paths.")
        sys.exit(1)

    # 2. PHASE 1: LOAD AND EXECUTE OUTER FUNCTION
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file {outer_path}: {e}")
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_outer_output = outer_data.get('output', None)

    print(f"Executing _get_image_quadrants with {len(outer_args)} args and {len(outer_kwargs)} kwargs.")
    try:
        # Run the agent function
        result_outer = _get_image_quadrants(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Execution of _get_image_quadrants failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. PHASE 2: VERIFICATION STRATEGY (Scenario A vs B)
    if inner_paths:
        # Scenario B: The outer result is expected to be a callable (Operator/Closure)
        print(f"Detected {len(inner_paths)} inner data files. Testing Factory/Closure pattern.")
        
        if not callable(result_outer):
            print(f"FAILURE: Inner data exists, expecting a callable return from outer function, but got {type(result_outer)}.")
            sys.exit(1)

        for i_path in inner_paths:
            print(f"  Testing inner file: {i_path}")
            try:
                with open(i_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                i_args = inner_data.get('args', [])
                i_kwargs = inner_data.get('kwargs', {})
                expected_inner_output = inner_data.get('output', None)

                # Execute the closure/operator
                actual_inner_output = result_outer(*i_args, **i_kwargs)

                # Verify
                passed, msg = recursive_check(expected_inner_output, actual_inner_output)
                if not passed:
                    print(f"  FAILURE on inner data {i_path}: {msg}")
                    sys.exit(1)
                else:
                    print(f"  Inner test passed.")

            except Exception as e:
                print(f"  Error processing inner file {i_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: The outer result is the final value
        print("No inner data files detected. Testing Direct Output pattern.")
        
        passed, msg = recursive_check(expected_outer_output, result_outer)
        if not passed:
            print(f"FAILURE: Output mismatch. {msg}")
            sys.exit(1)
        
    # Final Success
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    run_test()