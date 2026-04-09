import sys
import os
import dill
import torch
import numpy as np
import traceback
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

# --- Helpers needed for dill loading (if any global dependencies exist in serialized data) ---
try:
    import skimage.metrics
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

def run_test():
    """
    Unit test for evaluate_results based on serialized data.
    Strategies:
    1. Scenario A: Direct execution (One data file).
    2. Scenario B: Factory/Closure execution (Outer and Inner data files).
    """
    
    # 1. Setup Data Paths
    data_paths = ['/data/yjh/svmbir-master_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    outer_path = None
    inner_paths = []

    # Identify files
    for path in data_paths:
        if "parent_function" in path:
            inner_paths.append(path)
        else:
            outer_path = path

    if not outer_path:
        print("Error: No standard outer data file found (standard_data_evaluate_results.pkl).")
        sys.exit(1)

    print(f"Loading Outer Data: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Failed to load outer data: {e}")
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_outer_output = outer_data.get('output', None)

    # 2. Execute Outer Function
    print("Executing evaluate_results with outer arguments...")
    try:
        # NOTE: evaluate_results writes a file to disk by default. 
        # We might want to suppress or redirect this, but for exact reproduction, we let it run.
        # It creates 'reconstruction_result.png' which is a side effect.
        actual_operator = evaluate_results(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Execution of evaluate_results failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Determine Scenario (A or B)
    
    # Scenario B: Inner paths exist -> The result of outer was a callable (closure/factory)
    if inner_paths:
        print(f"Scenario B detected: {len(inner_paths)} inner data files found. Testing factory pattern.")
        
        if not callable(actual_operator):
            print("Error: Inner files exist, implying Factory pattern, but outer result is not callable.")
            sys.exit(1)

        for inner_path in inner_paths:
            print(f"  Testing Inner Data: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"  Failed to load inner data {inner_path}: {e}")
                continue

            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_inner_output = inner_data.get('output', None)

            try:
                actual_inner_result = actual_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"  Execution of inner function failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Verification for inner result
            passed, msg = recursive_check(expected_inner_output, actual_inner_result)
            if not passed:
                print(f"  FAILED: Inner comparison failed for {inner_path}")
                print(f"  Details: {msg}")
                sys.exit(1)
            else:
                print(f"  Inner Result Verified.")

    # Scenario A: No inner paths -> Simple function execution
    else:
        print("Scenario A detected: Direct function execution.")
        
        # Verify result directly against outer output
        passed, msg = recursive_check(expected_outer_output, actual_operator)
        if not passed:
            print("FAILED: Output comparison failed.")
            print(f"Details: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    run_test()