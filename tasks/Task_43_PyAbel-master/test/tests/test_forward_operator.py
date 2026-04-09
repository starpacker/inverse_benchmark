import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
try:
    from agent_forward_operator import forward_operator
except ImportError:
    print("Error: Could not import 'forward_operator' from 'agent_forward_operator.py'")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    print("Error: Could not import 'recursive_check' from 'verification_utils'")
    sys.exit(1)

def load_data(path):
    with open(path, 'rb') as f:
        return dill.load(f)

def run_test():
    # 1. Configuration
    data_paths = ['/data/yjh/PyAbel-master_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # 2. Identify Data Files
    # Based on the provided list, we only have the standard data file, no parent/inner files.
    outer_path = next((p for p in data_paths if 'standard_data_forward_operator.pkl' in p), None)
    inner_paths = [p for p in data_paths if 'standard_data_parent_function_forward_operator_' in p]
    
    if not outer_path:
        print("Error: Standard data file 'standard_data_forward_operator.pkl' not found.")
        sys.exit(1)

    # 3. Load Outer Data
    try:
        outer_data = load_data(outer_path)
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        outer_expected = outer_data.get('output')
    except Exception as e:
        print(f"Error loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Execute Logic
    try:
        # Scenario B detection: Check if the result of the outer call is expected to be a callable
        # or if we have inner paths to verify against.
        # However, looking at the provided code for forward_operator, it returns 'aim' (a numpy array),
        # not a function. It's a direct transform function (Scenario A).
        
        print("Executing forward_operator with loaded arguments...")
        actual_result = forward_operator(*outer_args, **outer_kwargs)

        # 5. Verification
        # If the function were a factory (Scenario B), actual_result would be a function, 
        # and we would need to run it against inner_data. 
        # Since forward_operator returns a result directly, we compare immediately.
        
        # However, just in case the provided 'gen_data_code' implies a decorator structure 
        # that might have captured a closure in some contexts, we handle the simple case first 
        # which matches the provided function signature.

        if inner_paths:
            # Scenario B logic (Unlikely based on provided function code, but robust for automation)
            print("Detected inner data files. Treating result as a callable operator...")
            if not callable(actual_result):
                print(f"Error: Expected a callable due to inner data presence, but got {type(actual_result)}")
                sys.exit(1)
            
            for inner_path in inner_paths:
                print(f"Testing against inner data: {inner_path}")
                inner_data = load_data(inner_path)
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                expected_inner_output = inner_data.get('output')
                
                inner_result = actual_result(*inner_args, **inner_kwargs)
                
                is_correct, msg = recursive_check(expected_inner_output, inner_result)
                if not is_correct:
                    print(f"FAILED on inner data {inner_path}: {msg}")
                    sys.exit(1)
        else:
            # Scenario A logic: Direct comparison
            print("No inner data found. Verifying direct output...")
            is_correct, msg = recursive_check(outer_expected, actual_result)
            if not is_correct:
                print(f"FAILED: {msg}")
                sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()