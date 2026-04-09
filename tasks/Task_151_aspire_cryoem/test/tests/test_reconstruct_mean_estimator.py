import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_reconstruct_mean_estimator import reconstruct_mean_estimator

# Import verification utility
from verification_utils import recursive_check

def main():
    """Main test function for reconstruct_mean_estimator."""
    
    # Define data paths
    data_paths = ['/data/yjh/aspire_cryoem_sandbox_sandbox/run_code/std_data/standard_data_reconstruct_mean_estimator.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_reconstruct_mean_estimator.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_reconstruct_mean_estimator.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run the function
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer function name: {outer_data.get('func_name', 'unknown')}")
    print(f"Number of args: {len(outer_args)}")
    print(f"Number of kwargs: {len(outer_kwargs)}")
    
    # Execute the target function
    print("Executing reconstruct_mean_estimator...")
    try:
        result = reconstruct_mean_estimator(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Function execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is a factory pattern (result is callable and we have inner data)
    if len(inner_paths) > 0 and callable(result):
        # Scenario B: Factory/Closure Pattern
        print("Detected factory pattern - processing inner data...")
        
        for inner_path in inner_paths:
            print(f"Loading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_output = inner_data.get('output', None)
            
            print(f"Inner function name: {inner_data.get('func_name', 'unknown')}")
            print("Executing returned operator...")
            
            try:
                actual_result = result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Inner function execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            print("Verifying results...")
            try:
                passed, msg = recursive_check(inner_output, actual_result)
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print(f"Inner test passed: {msg}")
        
        print("TEST PASSED")
        sys.exit(0)
    else:
        # Scenario A: Simple Function
        print("Detected simple function pattern...")
        
        # Verify results against outer output
        print("Verifying results...")
        try:
            passed, msg = recursive_check(outer_output, result)
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print(f"TEST PASSED: {msg}")
            sys.exit(0)

if __name__ == "__main__":
    main()