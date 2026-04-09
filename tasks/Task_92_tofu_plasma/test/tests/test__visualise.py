import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent__visualise import _visualise
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/tofu_plasma_sandbox_sandbox/run_code/std_data/standard_data__visualise.pkl']

def main():
    """Main test function for _visualise."""
    
    # Separate outer and inner paths
    outer_path = None
    inner_path = None
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_path = path
        elif basename == 'standard_data__visualise.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data__visualise.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    print(f"[INFO] Loaded outer data from: {outer_path}")
    print(f"[INFO] Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"[INFO] Number of args: {len(outer_args)}")
    print(f"[INFO] Kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the function
    try:
        result = _visualise(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute _visualise")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if result is callable (factory pattern)
    if callable(result) and inner_path is not None:
        # Scenario B: Factory/Closure Pattern
        print("[INFO] Detected factory pattern - result is callable")
        agent_operator = result
        
        # Load inner data
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load inner data from {inner_path}")
            print(f"Exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected_output = inner_data.get('output')
        
        print(f"[INFO] Loaded inner data from: {inner_path}")
        
        # Execute the operator
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute agent_operator")
            print(f"Exception: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function
        print("[INFO] Simple function pattern detected")
    
    # Phase 2: Verification
    try:
        passed, msg = recursive_check(expected_output, result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        print(f"Expected type: {type(expected_output)}")
        print(f"Actual type: {type(result)}")
        sys.exit(1)

if __name__ == "__main__":
    main()