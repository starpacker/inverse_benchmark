import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_forward_operator import forward_operator
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/GPRPy_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Classify paths into outer and inner
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_forward_operator.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}")
        print(traceback.format_exc())
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output')
    
    try:
        result = forward_operator(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute forward_operator with outer data")
        print(traceback.format_exc())
        sys.exit(1)
    
    # Check if this is Scenario B (factory/closure pattern) with inner data
    if len(inner_paths) > 0 and callable(result):
        # Scenario B: Factory pattern
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
            expected = inner_data.get('output')
            
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator with inner data from {inner_path}")
                print(traceback.format_exc())
                sys.exit(1)
            
            # Verify results
            try:
                passed, msg = recursive_check(expected, actual_result)
            except Exception as e:
                print(f"ERROR: Verification failed with exception")
                print(traceback.format_exc())
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED for inner data {inner_path}")
                print(f"Mismatch details: {msg}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    else:
        # Scenario A: Simple function
        expected = outer_output
        
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: Verification failed with exception")
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