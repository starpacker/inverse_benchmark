import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_project_points import project_points
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/myptv_sandbox_sandbox/run_code/std_data/standard_data_project_points.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_project_points.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_project_points.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute the function
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
        result = project_points(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute project_points with outer data")
        print(traceback.format_exc())
        sys.exit(1)
    
    # Check if we have inner data (Scenario B: Factory/Closure pattern)
    if inner_paths:
        # Scenario B: The result should be callable
        if not callable(result):
            print("ERROR: Expected a callable from project_points but got non-callable")
            sys.exit(1)
        
        agent_operator = result
        
        # Load inner data and execute
        inner_path = inner_paths[0]  # Use the first inner path
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
    else:
        # Scenario A: Simple function, result is the output
        expected = outer_output
    
    # Verification
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
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()