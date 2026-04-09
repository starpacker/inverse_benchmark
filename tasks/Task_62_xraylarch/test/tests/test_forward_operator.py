import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_forward_operator import forward_operator
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/xraylarch_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']

def main():
    """Main test function."""
    
    # Step 1: Analyze data paths to determine test strategy
    outer_path = None
    inner_path = None
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_path = path
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_forward_operator.pkl)")
        sys.exit(1)
    
    # Step 2: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Successfully loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract outer args and kwargs
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    
    # Step 3: Execute the target function
    try:
        result = forward_operator(*outer_args, **outer_kwargs)
        print("Successfully executed forward_operator")
    except Exception as e:
        print(f"ERROR: Failed to execute forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Determine expected output and compare
    if inner_path is not None:
        # Scenario B: Factory/Closure Pattern
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
        
        # The result from Phase 1 should be callable
        if not callable(result):
            print("ERROR: Expected forward_operator to return a callable (closure/operator)")
            sys.exit(1)
        
        try:
            actual_result = result(*inner_args, **inner_kwargs)
            print("Successfully executed the returned operator")
        except Exception as e:
            print(f"ERROR: Failed to execute the returned operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected = inner_data.get('output')
    else:
        # Scenario A: Simple Function
        actual_result = result
        expected = outer_data.get('output')
    
    # Step 5: Compare results
    try:
        passed, msg = recursive_check(expected, actual_result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()