import sys
import os
import dill
import traceback

# Add necessary paths
sys.path.insert(0, '/data/yjh/pycurious_sandbox_sandbox/run_code')
sys.path.insert(0, '/data/yjh/pycurious_sandbox_sandbox')

import numpy as np

# Import the target function
from agent_evaluate_results import evaluate_results

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for evaluate_results."""
    
    data_paths = ['/data/yjh/pycurious_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Separate outer and inner data paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_evaluate_results.pkl)")
        sys.exit(1)
    
    print(f"[INFO] Outer data path: {outer_path}")
    print(f"[INFO] Inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and run the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print("[INFO] Successfully loaded outer data")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    print(f"[INFO] Outer args count: {len(outer_args)}")
    print(f"[INFO] Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Scenario A: Simple function (no inner data)
    if len(inner_paths) == 0:
        print("[INFO] Scenario A: Simple function test")
        
        try:
            # Execute the function
            result = evaluate_results(*outer_args, **outer_kwargs)
            print("[INFO] Function executed successfully")
        except Exception as e:
            print(f"ERROR: Function execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Compare result with expected output
        try:
            passed, msg = recursive_check(expected_output, result)
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
    
    # Scenario B: Factory/Closure pattern (has inner data)
    else:
        print("[INFO] Scenario B: Factory/Closure pattern test")
        
        # Phase 1: Create the operator/closure
        try:
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            print("[INFO] Operator created successfully")
        except Exception as e:
            print(f"ERROR: Failed to create operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify that the result is callable
        if not callable(agent_operator):
            print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)
        
        # Phase 2: Load inner data and execute the operator
        for inner_path in inner_paths:
            print(f"[INFO] Processing inner data: {inner_path}")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print("[INFO] Successfully loaded inner data")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)
            
            print(f"[INFO] Inner args count: {len(inner_args)}")
            print(f"[INFO] Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("[INFO] Operator executed successfully")
            except Exception as e:
                print(f"ERROR: Operator execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare result with expected output
            try:
                passed, msg = recursive_check(inner_expected, result)
                if passed:
                    print(f"[INFO] Inner test passed for {os.path.basename(inner_path)}")
                else:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()