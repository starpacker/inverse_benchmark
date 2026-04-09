import sys
import os
import dill
import numpy as np
import traceback

# Handle optional torch dependency
try:
    import torch
except ImportError:
    torch = None

from agent_make_uniform_grid import make_uniform_grid, CartesianGrid, Grid, Field
from verification_utils import recursive_check

def run_test():
    # 1. Define Data Paths
    data_paths = ['/data/yjh/hcipy-master_sandbox/run_code/std_data/standard_data_make_uniform_grid.pkl']
    
    # 2. Strategy: Determine if this is a Closure/Factory or Simple Function
    outer_path = None
    inner_paths = []

    for p in data_paths:
        if 'parent_function_make_uniform_grid' in p:
            inner_paths.append(p)
        elif 'standard_data_make_uniform_grid.pkl' in p:
            outer_path = p
    
    if not outer_path:
        print("Test failed: Primary data file 'standard_data_make_uniform_grid.pkl' not found in paths.")
        sys.exit(1)

    print(f"Loading data from {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Test failed: Could not load data file. Error: {e}")
        sys.exit(1)

    # 3. Reconstruct the Operator/Result
    try:
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        
        # Execute the function under test
        print("Executing make_uniform_grid...")
        actual_result = make_uniform_grid(*args, **kwargs)
        
    except Exception as e:
        print("Test failed: Execution of make_uniform_grid raised an exception.")
        traceback.print_exc()
        sys.exit(1)

    # 4. Handle Scenarios
    
    # Scenario B: Closure/Factory Pattern (if inner files exist)
    if inner_paths:
        print(f"Detected Factory Pattern. {len(inner_paths)} inner data files found.")
        
        if not callable(actual_result):
            print("Test failed: Expected make_uniform_grid to return a callable (Operator), but got:", type(actual_result))
            sys.exit(1)
            
        # Iterate through inner calls
        for i_path in inner_paths:
            print(f"Testing inner execution with {i_path}")
            try:
                with open(i_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                expected_inner_output = inner_data.get('output')
                
                # Execute the closure
                actual_inner_output = actual_result(*inner_args, **inner_kwargs)
                
                # Verify
                passed, msg = recursive_check(expected_inner_output, actual_inner_output)
                if not passed:
                    print(f"Test failed for inner data {i_path}: {msg}")
                    sys.exit(1)
                    
            except Exception as e:
                print(f"Test failed during inner execution of {i_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

    # Scenario A: Simple Function (Direct Result)
    else:
        print("Detected Simple Function Pattern.")
        expected_result = outer_data.get('output')
        
        # In hcipy/physics, make_uniform_grid returns a Grid object. 
        # Grid objects might have complex internal states. recursive_check handles numpy arrays and basic types.
        # If the output is a custom object (like Grid), recursive_check might need to compare attributes.
        # However, usually standard recursive_check compares dicts/lists/primitives/arrays.
        
        # Special handling for Grid objects if recursive_check isn't specialized for them:
        # We ensure properties like coords and weights match.
        
        passed, msg = recursive_check(expected_result, actual_result)
        
        if not passed:
            # Fallback: Detailed debug for Grid objects if standard check fails on object identity
            if hasattr(actual_result, 'coords') and hasattr(expected_result, 'coords'):
                print("Detailed Grid check...")
                coords_match, c_msg = recursive_check(expected_result.coords, actual_result.coords)
                weights_match, w_msg = recursive_check(expected_result.weights, actual_result.weights)
                
                if coords_match and weights_match:
                    passed = True
                else:
                    msg = f"Grid Mismatch. Coords: {c_msg}, Weights: {w_msg}"

        if not passed:
            print(f"Test failed: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    run_test()