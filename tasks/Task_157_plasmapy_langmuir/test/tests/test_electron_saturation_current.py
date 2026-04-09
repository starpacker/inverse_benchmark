import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_electron_saturation_current import electron_saturation_current
from verification_utils import recursive_check

def main():
    # Define data paths
    data_paths = ['/data/yjh/plasmapy_langmuir_sandbox_sandbox/run_code/std_data/standard_data_electron_saturation_current.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_electron_saturation_current.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_electron_saturation_current.pkl)")
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
    outer_output = outer_data.get('output', None)
    
    # Phase 2: Execute the function
    try:
        result = electron_saturation_current(*outer_args, **outer_kwargs)
        print(f"Function executed successfully")
    except Exception as e:
        print(f"ERROR: Failed to execute electron_saturation_current: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if this is a factory pattern (Scenario B) or simple function (Scenario A)
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("Detected factory pattern - loading inner data")
        
        # The result should be callable
        if not callable(result):
            print(f"ERROR: Expected callable operator but got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Load inner data and execute
        inner_path = inner_paths[0]  # Use the first inner path
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
        expected = inner_data.get('output', None)
        
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
            print(f"Inner function executed successfully")
        except Exception as e:
            print(f"ERROR: Failed to execute inner function: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function
        print("Simple function pattern - using outer data for verification")
        expected = outer_output
    
    # Phase 3: Verification
    try:
        passed, msg = recursive_check(expected, result)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            print(f"Expected: {expected}")
            print(f"Actual: {result}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()