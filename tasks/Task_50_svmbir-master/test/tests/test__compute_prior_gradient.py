import sys
import os
import dill
import numpy as np
import traceback

# Add the directory containing the target function to the path
sys.path.append(os.path.dirname(__file__))

# Import the target function
from agent__compute_prior_gradient import _compute_prior_gradient
from verification_utils import recursive_check

def test_compute_prior_gradient():
    """
    Test script for _compute_prior_gradient.
    
    Strategy:
    1. Identify data files for inputs and expected outputs.
    2. Handle two scenarios:
       - Scenario A: The function returns a value directly.
       - Scenario B: The function returns a closure/factory (indicated by 'parent_function' files).
    3. Execute the function and compare results.
    """
    data_paths = ['/data/yjh/svmbir-master_sandbox/run_code/std_data/standard_data__compute_prior_gradient.pkl']
    
    # Filter paths
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        if "standard_data__compute_prior_gradient.pkl" in p:
            outer_path = p
        elif "parent_function" in p and "_compute_prior_gradient" in p:
            inner_paths.append(p)

    if not outer_path:
        print("Error: standard_data__compute_prior_gradient.pkl not found in provided paths.")
        sys.exit(1)

    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer data: {e}")
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_outer_output = outer_data.get('output', None)

    # --- Execution Step 1: Run the main function ---
    print("Running _compute_prior_gradient with loaded outer arguments...")
    try:
        # Note: _compute_prior_gradient uses a helper _qggmrf_derivative internally.
        # Since we imported _compute_prior_gradient, the helper defined in the same module should be accessible.
        result_step_1 = _compute_prior_gradient(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Branching Logic based on file structure ---
    
    if not inner_paths:
        # Scenario A: Simple function execution. 
        # The result of step 1 is the final result.
        print("Scenario A detected: Direct function execution.")
        actual_result = result_step_1
        expected_result = expected_outer_output
        
        passed, msg = recursive_check(expected_result, actual_result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
            
    else:
        # Scenario B: Factory/Closure pattern.
        # result_step_1 is likely a callable (operator) that needs to be invoked with inner data.
        print(f"Scenario B detected: Factory pattern with {len(inner_paths)} inner execution files.")
        
        if not callable(result_step_1):
            print("Error: Expected a callable return from outer function in Scenario B, but got:", type(result_step_1))
            sys.exit(1)
            
        all_passed = True
        
        for inner_path in inner_paths:
            print(f"  Processing inner file: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"  Error loading inner data {inner_path}: {e}")
                all_passed = False
                continue
                
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_inner_output = inner_data.get('output', None)
            
            try:
                actual_inner_result = result_step_1(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"  Inner execution failed for {inner_path}: {e}")
                traceback.print_exc()
                all_passed = False
                continue
                
            passed, msg = recursive_check(expected_inner_output, actual_inner_result)
            if not passed:
                print(f"  Comparison FAILED for {inner_path}: {msg}")
                all_passed = False
            else:
                print(f"  Comparison PASSED for {inner_path}")

        if all_passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print("TEST FAILED: One or more inner executions failed comparison.")
            sys.exit(1)

if __name__ == "__main__":
    test_compute_prior_gradient()