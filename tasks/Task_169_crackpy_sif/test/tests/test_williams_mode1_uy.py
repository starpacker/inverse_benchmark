import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_williams_mode1_uy import williams_mode1_uy
from verification_utils import recursive_check

def main():
    # Define data paths
    data_paths = ['/data/yjh/crackpy_sif_sandbox_sandbox/run_code/std_data/standard_data_williams_mode1_uy.pkl']
    
    # Filter paths to find outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_williams_mode1_uy.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_williams_mode1_uy.pkl)")
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
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    
    # Execute the function
    try:
        result = williams_mode1_uy(*outer_args, **outer_kwargs)
        print("Successfully executed williams_mode1_uy")
    except Exception as e:
        print(f"ERROR: Failed to execute williams_mode1_uy: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine if this is a factory pattern or simple function
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        # The result should be callable
        if not callable(result):
            print("ERROR: Expected a callable result for factory pattern, but got non-callable")
            sys.exit(1)
        
        agent_operator = result
        
        # Load inner data and execute
        inner_path = inner_paths[0]  # Use first inner path
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
        
        try:
            actual_result = agent_operator(*inner_args, **inner_kwargs)
            print("Successfully executed agent operator with inner args")
        except Exception as e:
            print(f"ERROR: Failed to execute agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected = inner_data.get('output')
    else:
        # Scenario A: Simple Function
        actual_result = result
        expected = outer_data.get('output')
    
    # Comparison
    try:
        passed, msg = recursive_check(expected, actual_result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if not passed:
        print(f"TEST FAILED: {msg}")
        print(f"Expected: {expected}")
        print(f"Actual: {actual_result}")
        sys.exit(1)
    else:
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()