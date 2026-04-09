import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_compute_cc import compute_cc
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/pysyd_astero_sandbox_sandbox/run_code/std_data/standard_data_compute_cc.pkl']
    
    # Classify data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_compute_cc.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_compute_cc.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    # Execute the function
    try:
        result = compute_cc(*outer_args, **outer_kwargs)
        print(f"Function executed successfully")
    except Exception as e:
        print(f"ERROR: Function execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is a factory pattern (result is callable and inner paths exist)
    if callable(result) and not isinstance(result, type) and len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("Detected factory pattern - result is callable and inner data exists")
        agent_operator = result
        
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output')
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Inner function executed successfully")
            except Exception as e:
                print(f"ERROR: Inner function execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify result
            try:
                passed, msg = recursive_check(expected_output, result)
                if not passed:
                    print(f"TEST FAILED (inner): {msg}")
                    sys.exit(1)
                print(f"Inner test passed")
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple Function
        print("Detected simple function pattern")
        
        # Verify result against expected output
        try:
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()