import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent__interpolate_image import _interpolate_image
from verification_utils import recursive_check

def main():
    """Main test function for _interpolate_image."""
    
    # Data paths provided
    data_paths = ['/data/yjh/pyidi_dic_sandbox_sandbox/run_code/std_data/standard_data__interpolate_image.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data__interpolate_image.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data__interpolate_image.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
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
    outer_output = outer_data.get('output')
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the function
    try:
        result = _interpolate_image(*outer_args, **outer_kwargs)
        print("Successfully executed _interpolate_image")
    except Exception as e:
        print(f"ERROR: Failed to execute _interpolate_image: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is a factory pattern (result is callable) and we have inner data
    if inner_paths and callable(result):
        # Scenario B: Factory/Closure Pattern
        print("Detected factory pattern - result is callable")
        agent_operator = result
        
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
            result = agent_operator(*inner_args, **inner_kwargs)
            print("Successfully executed agent_operator with inner data")
        except Exception as e:
            print(f"ERROR: Failed to execute agent_operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function
        print("Detected simple function pattern")
        expected = outer_output
    
    # Phase 2: Verification
    try:
        passed, msg = recursive_check(expected, result)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()