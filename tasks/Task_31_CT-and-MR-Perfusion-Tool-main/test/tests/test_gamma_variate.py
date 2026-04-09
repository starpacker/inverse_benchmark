import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_gamma_variate import gamma_variate
from verification_utils import recursive_check

def test_gamma_variate():
    """
    Test script for gamma_variate using serialized data.
    Handles both direct execution (Scenario A) and factory pattern (Scenario B).
    """
    data_paths = ['/data/yjh/CT-and-MR-Perfusion-Tool-main_sandbox/run_code/std_data/standard_data_gamma_variate.pkl']
    
    # 1. Identify Data Files
    outer_data_path = None
    inner_data_path = None

    for path in data_paths:
        if 'standard_data_gamma_variate.pkl' in path:
            outer_data_path = path
        elif 'parent_function_gamma_variate' in path:
            inner_data_path = path

    if not outer_data_path:
        print("Error: Standard outer data file not found in paths.")
        sys.exit(1)

    print(f"Loading Outer Data from: {outer_data_path}")
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer data: {e}")
        sys.exit(1)

    # 2. Execute Outer Function (Reconstruction)
    try:
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        
        # Run the function to be tested
        actual_result_outer = gamma_variate(*outer_args, **outer_kwargs)
        
    except Exception as e:
        print(f"Error executing gamma_variate: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Determine Verification Strategy based on existence of Inner Data
    if inner_data_path:
        # Scenario B: Factory Pattern (Function returns a function)
        print(f"Factory Pattern detected. Loading Inner Data from: {inner_data_path}")
        
        if not callable(actual_result_outer):
            print(f"Error: Expected gamma_variate to return a callable (factory pattern), but got {type(actual_result_outer)}")
            sys.exit(1)

        try:
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"Error loading inner data: {e}")
            sys.exit(1)

        try:
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_result = inner_data.get('output')

            # Execute the closure/operator returned by the outer function
            actual_result = actual_result_outer(*inner_args, **inner_kwargs)
            
        except Exception as e:
            print(f"Error executing inner closure function: {e}")
            traceback.print_exc()
            sys.exit(1)

    else:
        # Scenario A: Direct Execution (Function returns a value)
        print("Direct Execution detected.")
        actual_result = actual_result_outer
        expected_result = outer_data.get('output')

    # 4. Verify Results
    print("Verifying results...")
    is_correct, msg = recursive_check(actual_result, expected_result)

    if is_correct:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    test_gamma_variate()