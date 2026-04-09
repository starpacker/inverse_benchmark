import sys
import os
import dill
import numpy as np
import traceback

# Add the path to find the module
sys.path.insert(0, '/data/yjh/mountainsort_spike_sandbox_sandbox')

from agent_compute_pred_templates import compute_pred_templates
from verification_utils import recursive_check

def main():
    """Main test function for compute_pred_templates."""
    
    data_paths = ['/data/yjh/mountainsort_spike_sandbox_sandbox/run_code/std_data/standard_data_compute_pred_templates.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        else:
            outer_path = path
    
    if outer_path is None:
        print("ERROR: No outer data file found (standard_data_compute_pred_templates.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run function
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
    expected_output = outer_data.get('output', None)
    
    print(f"Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"Number of args: {len(outer_args)}")
    print(f"Kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the function
    try:
        result = compute_pred_templates(*outer_args, **outer_kwargs)
        print("Function executed successfully")
    except Exception as e:
        print(f"ERROR: Function execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if result is callable (factory pattern) and we have inner paths
    if callable(result) and len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("Detected factory pattern - result is callable")
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
            expected_output = inner_data.get('output', None)
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Inner function executed successfully")
            except Exception as e:
                print(f"ERROR: Inner function execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            try:
                passed, msg = recursive_check(expected_output, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                print(f"Verification passed for inner data: {inner_path}")
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function - compare result directly
        print("Simple function pattern - comparing output directly")
        
        try:
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()