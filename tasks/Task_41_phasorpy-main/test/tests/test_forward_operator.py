import sys
import os
import dill
import numpy as np
import traceback

# Handle optional torch import to prevent ModuleNotFoundError if not installed
try:
    import torch
except ImportError:
    torch = None

from agent_forward_operator import forward_operator
from verification_utils import recursive_check

def test_forward_operator():
    """
    Unit test for forward_operator.
    Strategy:
    1. Detects if this is a simple function call or a factory pattern based on file existence.
    2. Loads data using dill.
    3. Executes the function.
    4. Compares results using recursive_check.
    """
    
    # 1. Define Data Paths
    # The instruction provided a specific list of paths.
    data_paths = ['/data/yjh/phasorpy-main_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Separate paths into outer (main function) and inner (result of factory if applicable)
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if 'standard_data_forward_operator.pkl' in path:
            outer_path = path
        elif 'parent_function_forward_operator' in path:
            inner_paths.append(path)
            
    if not outer_path:
        print("Skipping test: standard_data_forward_operator.pkl not found in provided paths.")
        sys.exit(0)

    # 2. Load Outer Data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Failed to load outer data from {outer_path}: {e}")
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_outer_output = outer_data.get('output')

    print(f"Loaded outer data. Args len: {len(outer_args)}, Kwargs keys: {list(outer_kwargs.keys())}")

    # 3. Execute Main Function
    try:
        # Run the function
        actual_result_or_operator = forward_operator(*outer_args, **outer_kwargs)
    except Exception as e:
        print("Execution of forward_operator failed:")
        traceback.print_exc()
        sys.exit(1)

    # 4. Scenario Determination & Verification
    
    # Scenario A: Simple Function (No inner paths found)
    if not inner_paths:
        print("Scenario A: Direct function execution.")
        
        # Verify result directly
        passed, msg = recursive_check(expected_outer_output, actual_result_or_operator)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

    # Scenario B: Factory Pattern (Inner paths exist)
    else:
        print("Scenario B: Factory/Closure pattern detected.")
        
        # In this scenario, 'actual_result_or_operator' is expected to be a callable (the operator)
        if not callable(actual_result_or_operator):
            print(f"TEST FAILED: Expected a callable operator, got {type(actual_result_or_operator)}")
            sys.exit(1)
            
        agent_operator = actual_result_or_operator
        
        # Iterate through inner data files (could be multiple calls to the generated operator)
        all_passed = True
        
        for inner_path in inner_paths:
            print(f"Testing against inner data: {os.path.basename(inner_path)}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"Failed to load inner data {inner_path}: {e}")
                all_passed = False
                continue
                
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_inner_output = inner_data.get('output')
            
            try:
                # Execute the operator generated in step 3
                actual_inner_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"Execution of generated operator failed for {inner_path}:")
                traceback.print_exc()
                all_passed = False
                continue
                
            passed, msg = recursive_check(expected_inner_output, actual_inner_result)
            if not passed:
                print(f"Comparison failed for {inner_path}: {msg}")
                all_passed = False
            else:
                print(f"Verification passed for {inner_path}")
        
        if all_passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print("TEST FAILED: One or more inner verifications failed.")
            sys.exit(1)

if __name__ == "__main__":
    test_forward_operator()