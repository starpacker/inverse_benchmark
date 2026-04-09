import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent__image_gradients import _image_gradients
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/pyidi_dic_sandbox_sandbox/run_code/std_data/standard_data__image_gradients.pkl']

def main():
    """Main test function."""
    
    # Filter data paths to find outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data__image_gradients.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data__image_gradients.pkl)")
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
    
    # Execute the function with outer args
    try:
        result = _image_gradients(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute _image_gradients with outer args")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is Scenario B (factory/closure pattern) or Scenario A (simple function)
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        # The result should be callable
        if not callable(result):
            print(f"ERROR: Expected callable result for factory pattern, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Process each inner path
        for inner_path in inner_paths:
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
            expected = inner_data.get('output')
            
            # Execute the operator with inner args
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator with inner args")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            passed, msg = recursive_check(expected, actual_result)
            if not passed:
                print(f"TEST FAILED for inner path: {inner_path}")
                print(f"Failure message: {msg}")
                sys.exit(1)
    else:
        # Scenario A: Simple function - compare result directly with outer output
        expected = outer_data.get('output')
        
        # Compare results
        passed, msg = recursive_check(expected, result)
        if not passed:
            print("TEST FAILED")
            print(f"Failure message: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()