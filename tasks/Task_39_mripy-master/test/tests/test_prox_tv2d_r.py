import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_prox_tv2d_r import prox_tv2d_r
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/mripy-master_sandbox/run_code/std_data/standard_data_prox_tv2d_r.pkl']
    
    # 1. Identify Data Files
    outer_path = None
    inner_path = None

    for p in data_paths:
        if 'parent_function_prox_tv2d_r' in p:
            inner_path = p
        elif 'standard_data_prox_tv2d_r.pkl' in p:
            outer_path = p

    if not outer_path:
        print("Error: standard_data_prox_tv2d_r.pkl not found in data_paths.")
        sys.exit(1)

    print(f"Loading Outer Data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer data: {e}")
        sys.exit(1)

    # 2. Reconstruct / Execute
    # The generation code suggests prox_tv2d_r might return a result directly (standard function)
    # OR return a callable (factory). We check the output type or existence of inner data.
    
    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_outer_output = outer_data.get('output', None)

    try:
        print("Executing prox_tv2d_r with outer arguments...")
        result_object = prox_tv2d_r(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Verification Logic
    if inner_path:
        # Scenario B: Factory Pattern
        if not callable(result_object):
            print(f"Error: Expected prox_tv2d_r to return a callable (factory pattern) because inner data exists, but got {type(result_object)}.")
            sys.exit(1)
            
        print(f"Loading Inner Data from: {inner_path}")
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"Error loading inner data: {e}")
            sys.exit(1)

        inner_args = inner_data.get('args', [])
        inner_kwargs = inner_data.get('kwargs', {})
        expected_inner_output = inner_data.get('output', None)

        try:
            print("Executing inner callable...")
            final_result = result_object(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"Inner execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
            
        print("Verifying Inner Result...")
        is_correct, msg = recursive_check(expected_inner_output, final_result)
        
    else:
        # Scenario A: Standard Function
        print("No inner data found. Verifying direct output...")
        final_result = result_object
        is_correct, msg = recursive_check(expected_outer_output, final_result)

    if is_correct:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()