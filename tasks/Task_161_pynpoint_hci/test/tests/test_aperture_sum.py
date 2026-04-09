import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_aperture_sum import aperture_sum
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/pynpoint_hci_sandbox_sandbox/run_code/std_data/standard_data_aperture_sum.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename:
            inner_paths.append(path)
        elif filename == 'standard_data_aperture_sum.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_aperture_sum.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute the function
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
    
    # Execute the function
    try:
        result = aperture_sum(*outer_args, **outer_kwargs)
        print("Successfully executed aperture_sum")
    except Exception as e:
        print(f"ERROR: Failed to execute aperture_sum: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine which scenario we're in
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        # The result should be callable, and we need to execute it with inner data
        if not callable(result):
            print("ERROR: Expected callable result for factory pattern, but got non-callable")
            sys.exit(1)
        
        agent_operator = result
        
        # Load inner data and execute
        inner_path = inner_paths[0]  # Take the first inner path
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
        
        try:
            actual_result = agent_operator(*inner_args, **inner_kwargs)
            print("Successfully executed agent_operator with inner data")
        except Exception as e:
            print(f"ERROR: Failed to execute agent_operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected = inner_data['output']
    else:
        # Scenario A: Simple function
        actual_result = result
        expected = outer_data['output']
    
    # Comparison
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