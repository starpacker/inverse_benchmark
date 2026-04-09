import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_pad_array import pad_array
from verification_utils import recursive_check


def main():
    # Data paths provided
    data_paths = ['/home/yjh/lensless_dl_sandbox/run_code/std_data/standard_data_pad_array.pkl']
    
    # Categorize paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_pad_array.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_pad_array.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    # Execute the function
    try:
        result = pad_array(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute pad_array with outer data")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is Scenario B (factory/closure pattern)
    if inner_paths:
        # Scenario B: The result should be a callable operator
        if not callable(result):
            print("ERROR: Expected pad_array to return a callable operator for Scenario B")
            print(f"Got type: {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Process each inner data file
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)
            
            try:
                inner_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator with inner data from {inner_path}")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify the result
            try:
                passed, msg = recursive_check(inner_expected, inner_result)
            except Exception as e:
                print(f"ERROR: Verification failed for inner data from {inner_path}")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED for inner data: {inner_path}")
                print(f"Verification message: {msg}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple function - compare result directly with expected output
        try:
            passed, msg = recursive_check(expected_output, result)
        except Exception as e:
            print(f"ERROR: Verification failed")
            print(f"Exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print("TEST FAILED")
            print(f"Verification message: {msg}")
            sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()