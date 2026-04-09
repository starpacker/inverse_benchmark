import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/mountainsort_spike_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: No outer data file found for load_and_preprocess_data")
        sys.exit(1)
    
    # Load outer data
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
    
    # Scenario A: Simple Function (no inner paths)
    if not inner_paths:
        print("Scenario A: Simple function test")
        try:
            result = load_and_preprocess_data(*outer_args, **outer_kwargs)
            print("Function executed successfully")
        except Exception as e:
            print(f"ERROR: Function execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Compare result with expected output
        try:
            passed, msg = recursive_check(outer_output, result)
            if passed:
                print("TEST PASSED")
                sys.exit(0)
            else:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Scenario B: Factory/Closure Pattern
    else:
        print("Scenario B: Factory/Closure pattern test")
        
        # Phase 1: Reconstruct operator
        try:
            agent_operator = load_and_preprocess_data(*outer_args, **outer_kwargs)
            print("Operator created successfully")
        except Exception as e:
            print(f"ERROR: Failed to create operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not callable(agent_operator):
            print("ERROR: Created operator is not callable")
            sys.exit(1)
        
        # Phase 2: Load inner data and execute
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
            inner_output = inner_data.get('output')
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Inner function executed successfully")
            except Exception as e:
                print(f"ERROR: Inner function execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare result with expected output
            try:
                passed, msg = recursive_check(inner_output, result)
                if passed:
                    print(f"TEST PASSED for {inner_path}")
                else:
                    print(f"TEST FAILED for {inner_path}: {msg}")
                    sys.exit(1)
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()