import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_triangulate_dlt import triangulate_dlt
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/myptv_sandbox_sandbox/run_code/std_data/standard_data_triangulate_dlt.pkl']

def main():
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_triangulate_dlt.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_triangulate_dlt.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}")
        print(traceback.format_exc())
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    # Execute the target function
    try:
        result = triangulate_dlt(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute triangulate_dlt")
        print(traceback.format_exc())
        sys.exit(1)
    
    # Check if result is callable (factory pattern) and we have inner paths
    if callable(result) and not isinstance(result, np.ndarray) and len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
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
            inner_expected = inner_data.get('output')
            
            try:
                inner_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator with inner data")
                print(traceback.format_exc())
                sys.exit(1)
            
            # Verify inner result
            try:
                passed, msg = recursive_check(inner_expected, inner_result)
            except Exception as e:
                print(f"ERROR: Verification failed with exception")
                print(traceback.format_exc())
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED for inner path {inner_path}")
                print(msg)
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    else:
        # Scenario A: Simple Function - compare result directly with expected output
        try:
            passed, msg = recursive_check(expected_output, result)
        except Exception as e:
            print(f"ERROR: Verification failed with exception")
            print(traceback.format_exc())
            sys.exit(1)
        
        if not passed:
            print("TEST FAILED")
            print(msg)
            sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()