import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results
# Import verification utility
from verification_utils import recursive_check

def test_evaluate_results():
    """
    Test script for evaluate_results function.
    Scenario: Simple Function Execution (Scenario A).
    """
    
    # 1. Define Data Paths
    data_paths = ['/data/yjh/us-beamform-linarray-master_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    outer_path = None
    inner_path = None

    # Classify paths
    for path in data_paths:
        if 'standard_data_evaluate_results.pkl' in path:
            outer_path = path
        elif 'standard_data_parent_function_evaluate_results' in path:
            inner_path = path

    print(f"Outer Data Path: {outer_path}")
    print(f"Inner Data Path: {inner_path}")

    if not outer_path:
        print("Error: Standard data file (outer) not found.")
        sys.exit(1)

    # 2. Phase 1: Load Outer Data and Execute Target
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print(f"Executing evaluate_results with {len(outer_args)} args and {len(outer_kwargs)} kwargs...")
        
        # Execute the function
        actual_result = evaluate_results(*outer_args, **outer_kwargs)
        
    except Exception as e:
        print(f"Execution failed during Phase 1: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Phase 2: Handle Closure/Factory Pattern if applicable
    # In this specific case, based on the provided paths, we only have outer data.
    # However, we check logic for Scenario B (Factory) just in case logic changes, 
    # but primarily proceed with Scenario A logic here.

    if inner_path:
        print("Detected Factory/Closure pattern (Scenario B).")
        if not callable(actual_result):
            print(f"Error: Expected a callable (operator) from Phase 1, but got {type(actual_result)}.")
            sys.exit(1)
            
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            # Update expected output to be the result of the inner execution
            expected_output = inner_data.get('output')
            
            print(f"Executing generated operator with {len(inner_args)} args and {len(inner_kwargs)} kwargs...")
            actual_result = actual_result(*inner_args, **inner_kwargs)
            
        except Exception as e:
            print(f"Execution failed during Phase 2 (Inner): {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        print("Standard Function Execution (Scenario A).")

    # 4. Verification
    print("\nVerifying results...")
    passed, msg = recursive_check(expected_output, actual_result)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        # Debugging info
        print(f"Expected Type: {type(expected_output)}")
        print(f"Actual Type: {type(actual_result)}")
        sys.exit(1)

if __name__ == "__main__":
    test_evaluate_results()