import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent__rescale_to_gt import _rescale_to_gt
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/asteroid_bss_sandbox_sandbox/run_code/std_data/standard_data__rescale_to_gt.pkl']

def main():
    # Determine outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data__rescale_to_gt.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data__rescale_to_gt.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and call the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}")
        print(traceback.format_exc())
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    try:
        result = _rescale_to_gt(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute _rescale_to_gt with outer args")
        print(traceback.format_exc())
        sys.exit(1)
    
    # Check if we have inner paths (Scenario B: factory/closure pattern)
    if inner_paths:
        # Scenario B: The result should be callable, and we need to use inner data
        if not callable(result):
            print("ERROR: Expected callable result from _rescale_to_gt (factory pattern), but got non-callable")
            sys.exit(1)
        
        agent_operator = result
        
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}")
                print(traceback.format_exc())
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator with inner args")
                print(traceback.format_exc())
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected, actual_result)
            except Exception as e:
                print(f"ERROR: recursive_check failed")
                print(traceback.format_exc())
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED for inner data {inner_path}")
                print(f"Mismatch details: {msg}")
                sys.exit(1)
    else:
        # Scenario A: Simple function, compare result directly with outer_output
        expected = outer_output
        
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: recursive_check failed")
            print(traceback.format_exc())
            sys.exit(1)
        
        if not passed:
            print("TEST FAILED")
            print(f"Mismatch details: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()