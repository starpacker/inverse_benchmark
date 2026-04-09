import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_compute_relative_error import compute_relative_error

# Import verification utility
from verification_utils import recursive_check

def main():
    # Define data paths
    data_paths = ['/data/yjh/hippylib_bayesian_sandbox_sandbox/run_code/std_data/standard_data_compute_relative_error.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_compute_relative_error.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_compute_relative_error.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Successfully loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    print(f"Outer args: {len(outer_args)} arguments")
    print(f"Outer kwargs: {list(outer_kwargs.keys())}")
    
    # Execute the function
    try:
        result = compute_relative_error(*outer_args, **outer_kwargs)
        print(f"Successfully executed compute_relative_error")
    except Exception as e:
        print(f"ERROR: Failed to execute compute_relative_error: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario based on inner paths
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print("Detected Scenario B: Factory/Closure Pattern")
        
        # Check if result is callable (an operator)
        if not callable(result):
            print(f"ERROR: Expected callable operator, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Load and execute inner data
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Successfully loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output')
            
            try:
                inner_result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Successfully executed agent_operator")
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify inner result
            passed, msg = recursive_check(inner_expected, inner_result)
            if not passed:
                print(f"TEST FAILED (inner execution): {msg}")
                sys.exit(1)
            print(f"Inner execution verification passed")
    else:
        # Scenario A: Simple Function
        print("Detected Scenario A: Simple Function")
        
        # Verify result against expected output
        passed, msg = recursive_check(expected_output, result)
        if not passed:
            print(f"TEST FAILED: {msg}")
            print(f"Expected: {expected_output}")
            print(f"Got: {result}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()