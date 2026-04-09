import sys
import os
import dill
import numpy as np
import traceback

# Add current directory to path so we can import the target module
sys.path.append(os.getcwd())

try:
    from agent_minus_log import minus_log
    from verification_utils import recursive_check
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def test_minus_log():
    data_paths = ['/data/yjh/tomopy-master_sandbox/run_code/std_data/standard_data_minus_log.pkl']
    
    # 1. Identify File Types
    outer_path = None
    inner_paths = []

    for path in data_paths:
        if 'parent_function_minus_log' in path:
            inner_paths.append(path)
        elif 'standard_data_minus_log.pkl' in path:
            outer_path = path

    if not outer_path:
        print("Error: standard_data_minus_log.pkl not found in provided paths.")
        sys.exit(1)

    # 2. Load Outer Data (The Factory/Direct Call)
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading {outer_path}: {e}")
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_outer_output = outer_data.get('output', None)

    print(f"Executing minus_log with loaded args/kwargs from {os.path.basename(outer_path)}...")
    
    try:
        # Run the function
        actual_result = minus_log(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Determine Verification Strategy
    # The presence of inner_paths usually implies the result is a closure/function (Scenario B),
    # but based on the provided paths and function code, minus_log seems to return a direct result (Scenario A).
    # However, we must handle both possibilities robustly.

    if inner_paths:
        # Scenario B: minus_log returned a callable (operator) that needs to be tested
        if not callable(actual_result):
            print(f"Error: Expected minus_log to return a callable (because inner data exists), but got {type(actual_result)}")
            sys.exit(1)
        
        print("Function returned a callable. Proceeding to test inner executions...")
        
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"Error loading inner data {inner_path}: {e}")
                continue

            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_inner_output = inner_data.get('output', None)

            print(f"  Running inner callable with args from {os.path.basename(inner_path)}...")
            try:
                inner_result = actual_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"  Inner execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            passed, msg = recursive_check(expected_inner_output, inner_result)
            if not passed:
                print(f"  Comparison failed for {os.path.basename(inner_path)}: {msg}")
                sys.exit(1)
            else:
                print(f"  Verification passed for {os.path.basename(inner_path)}")

    else:
        # Scenario A: minus_log returned the final result directly
        print("Direct result verification...")
        passed, msg = recursive_check(expected_outer_output, actual_result)
        if not passed:
            print(f"Comparison failed: {msg}")
            sys.exit(1)
        else:
            print("Verification passed.")

    print("TEST PASSED")

if __name__ == "__main__":
    test_minus_log()