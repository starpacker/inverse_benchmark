import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_resize_array import resize_array
from verification_utils import recursive_check


def main():
    """Main test function for resize_array"""
    
    data_paths = ['/home/yjh/dpi_task2_sandbox/run_code/std_data/standard_data_resize_array.pkl']
    
    # Analyze data paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_resize_array.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find standard_data_resize_array.pkl")
        sys.exit(1)
    
    # Phase 1: Load outer data and run function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    
    try:
        result = resize_array(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute resize_array with outer data")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if this is Scenario A or B
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        # The result should be callable
        if not callable(result):
            print("ERROR: Expected resize_array to return a callable (closure/operator)")
            print(f"Got type: {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Process each inner path
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
            expected = inner_data.get('output')
            
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator with inner data")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            try:
                passed, msg = recursive_check(expected, actual_result)
            except Exception as e:
                print(f"ERROR: recursive_check raised an exception")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED for inner path: {inner_path}")
                print(f"Verification message: {msg}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple function - compare result directly
        expected = outer_data.get('output')
        
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: recursive_check raised an exception")
            print(f"Exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print("TEST FAILED")
            print(f"Verification message: {msg}")
            sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()