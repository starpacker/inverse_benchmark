import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

def main():
    """Main test function for load_and_preprocess_data."""
    
    # Data paths provided
    data_paths = ['/data/yjh/rmtools_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Classify paths into outer and inner
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
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
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the function
    try:
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Successfully executed load_and_preprocess_data")
    except Exception as e:
        print(f"ERROR: Failed to execute load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is a factory pattern (result is callable and inner paths exist)
    if inner_paths and callable(result):
        # Scenario B: Factory/Closure Pattern
        print("Detected factory/closure pattern")
        agent_operator = result
        
        for inner_path in inner_paths:
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
            expected_output = inner_data.get('output')
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed inner operator")
            except Exception as e:
                print(f"ERROR: Failed to execute inner operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            try:
                passed, msg = recursive_check(expected_output, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                print(f"Inner test passed for: {inner_path}")
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    else:
        # Scenario A: Simple function - compare result directly
        print("Detected simple function pattern")
        
        try:
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            print("TEST PASSED")
            sys.exit(0)
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()