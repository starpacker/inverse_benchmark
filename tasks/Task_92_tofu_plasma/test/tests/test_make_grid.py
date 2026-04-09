import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_make_grid import make_grid

# Import verification utility
from verification_utils import recursive_check

def main():
    """Main test function for make_grid."""
    
    # Data paths provided
    data_paths = ['/data/yjh/tofu_plasma_sandbox_sandbox/run_code/std_data/standard_data_make_grid.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        else:
            outer_path = path
    
    # Phase 1: Load outer data and execute the function
    if outer_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)
    
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}")
        print(traceback.format_exc())
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    
    try:
        agent_result = make_grid(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute make_grid with outer data")
        print(traceback.format_exc())
        sys.exit(1)
    
    # Check if this is a factory/closure pattern (Scenario B) or simple function (Scenario A)
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        # agent_result should be callable
        if not callable(agent_result):
            print("ERROR: Expected callable from make_grid but got non-callable")
            sys.exit(1)
        
        agent_operator = agent_result
        
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
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator with inner data")
                print(traceback.format_exc())
                sys.exit(1)
            
            # Verify results
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"ERROR: Verification failed with exception")
                print(traceback.format_exc())
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
    else:
        # Scenario A: Simple Function
        expected = outer_data.get('output')
        result = agent_result
        
        # Verify results
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: Verification failed with exception")
            print(traceback.format_exc())
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()