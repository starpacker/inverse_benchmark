import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_compute_dynamic_range import compute_dynamic_range
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/priism_radio_sandbox_sandbox/run_code/std_data/standard_data_compute_dynamic_range.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_compute_dynamic_range.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_compute_dynamic_range.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
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
        agent_result = compute_dynamic_range(*outer_args, **outer_kwargs)
        print(f"Function executed successfully")
    except Exception as e:
        print(f"ERROR: Failed to execute compute_dynamic_range: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine if this is Scenario A or B
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        # The result should be callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable operator but got {type(agent_result)}")
            sys.exit(1)
        
        # Load inner data and execute
        inner_path = inner_paths[0]  # Use first inner path
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
        expected = inner_data.get('output')
        
        try:
            result = agent_result(*inner_args, **inner_kwargs)
            print(f"Inner function executed successfully")
        except Exception as e:
            print(f"ERROR: Failed to execute inner function: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function
        result = agent_result
        expected = outer_data.get('output')
    
    # Comparison
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if not passed:
        print(f"TEST FAILED: {msg}")
        print(f"Expected: {expected}")
        print(f"Got: {result}")
        sys.exit(1)
    else:
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()