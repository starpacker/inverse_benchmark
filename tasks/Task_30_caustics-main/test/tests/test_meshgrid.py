import sys
import os
import dill
import torch
import numpy as np
import traceback
from agent_meshgrid import meshgrid
from verification_utils import recursive_check

def test_meshgrid():
    """
    Test script for the meshgrid function.
    
    Strategy:
    1. Identify data files provided in `data_paths`.
    2. Load the main data file `standard_data_meshgrid.pkl`.
    3. Since meshgrid is a standard function (not a closure factory based on the source code),
       we treat it as Scenario A: Load args -> Run -> Compare.
    4. However, we check if the result is callable (Scenario B detection) just in case dynamic behavior occurred.
    """
    
    data_paths = ['/data/yjh/caustics-main_sandbox/run_code/std_data/standard_data_meshgrid.pkl']
    
    # Check for inner data files which would imply a factory pattern (Scenario B)
    outer_path = None
    inner_path = None
    
    for path in data_paths:
        if "standard_data_meshgrid.pkl" in path:
            outer_path = path
        elif "parent_function_meshgrid" in path:
            inner_path = path
            
    if not outer_path:
        print("Error: standard_data_meshgrid.pkl not found in paths.")
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
    expected_result = outer_data.get('output', None)

    print(f"Executing meshgrid with args: {len(outer_args)} items, kwargs: {list(outer_kwargs.keys())}")
    
    try:
        # Phase 1: Run the function
        actual_result = meshgrid(*outer_args, **outer_kwargs)
        
        # Phase 2: Handle potential Factory Pattern (Closure)
        # If the result is callable and we have an inner data file, we proceed to execute the result.
        if callable(actual_result) and not isinstance(actual_result, (torch.Tensor, tuple, list)):
            if inner_path:
                print(f"Result is callable (Factory detected). Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                expected_result = inner_data.get('output', None) # Update expected to the inner output
                
                print("Executing inner callable...")
                actual_result = actual_result(*inner_args, **inner_kwargs)
            else:
                # Result is callable but no inner data provided.
                # This often happens if the 'output' in the outer pickle IS the callable itself.
                # We can't verify execution, but we can verify it matches the structure if possible,
                # or strictly compare it against the expected object if it was serialized.
                pass 
        
        # Phase 3: Verification
        print("Verifying results...")
        passed, msg = recursive_check(expected_result, actual_result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_meshgrid()