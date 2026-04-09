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
    data_paths = ['/data/yjh/pycurious_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Identify outer and inner paths
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
    
    # Phase 1: Load outer data and execute forward_operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output')
    
    try:
        result = forward_operator(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if we have inner data (Scenario B: factory/closure pattern)
    if inner_paths:
        # Scenario B: The result should be callable, and we need to test it with inner data
        if not callable(result):
            print("ERROR: Expected forward_operator to return a callable, but it didn't")
            sys.exit(1)
        
        agent_operator = result
        
        # Load inner data and execute the operator
        inner_path = inner_paths[0]  # Use the first inner path
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load inner data file: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected = inner_data.get('output')
        
        try:
            actual_result = agent_operator(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute the returned operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify results
        try:
            passed, msg = recursive_check(expected, actual_result)
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)
    else:
        # Scenario A: Simple function - compare result directly with outer_output
        expected = outer_output
        
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)

if __name__ == "__main__":
    main()