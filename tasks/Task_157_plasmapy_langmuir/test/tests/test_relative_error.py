import sys
import os
import dill
import traceback

# Import the target function
from agent_relative_error import relative_error
from verification_utils import recursive_check

# Try importing optional modules
try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

def main():
    # Data paths provided
    data_paths = ['/data/yjh/plasmapy_langmuir_sandbox_sandbox/run_code/std_data/standard_data_relative_error.pkl']
    
    # Filter paths to identify outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_relative_error.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: No outer data file found (standard_data_relative_error.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute the function
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
    expected_output = outer_data.get('output', None)
    
    try:
        # Execute the target function with outer args
        agent_result = relative_error(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute relative_error with outer args")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if we have inner data (factory/closure pattern)
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        # The result from Phase 1 should be callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable from relative_error, got {type(agent_result)}")
            sys.exit(1)
        
        agent_operator = agent_result
        
        # Process first inner path (or could iterate if multiple)
        inner_path = inner_paths[0]
        
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
        expected = inner_data.get('output', None)
        
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute agent_operator with inner args")
            print(f"Exception: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function
        result = agent_result
        expected = expected_output
    
    # Phase 3: Verification
    try:
        passed, msg = recursive_check(expected, result)
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
        print(f"Expected: {expected}")
        print(f"Got: {result}")
        sys.exit(1)

if __name__ == "__main__":
    main()