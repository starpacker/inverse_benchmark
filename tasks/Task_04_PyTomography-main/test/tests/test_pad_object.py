import sys
import os
import dill
import torch
import numpy as np
import traceback

# Ensure the module under test is importable
# Assuming agent_pad_object.py is in the Python path or current directory
try:
    from agent_pad_object import pad_object
except ImportError:
    # If the file is in the current directory but not importable, try adding valid paths
    sys.path.append(os.getcwd())
    try:
        from agent_pad_object import pad_object
    except ImportError:
        print("CRITICAL: Could not import 'pad_object' from 'agent_pad_object'.")
        sys.exit(1)

from verification_utils import recursive_check

# Data paths provided in instructions
data_paths = ['/data/yjh/PyTomography-main_sandbox/run_code/std_data/standard_data_pad_object.pkl']

def main():
    try:
        # 1. DATA FILE ANALYSIS
        # We analyze the paths to determine if we are in Scenario A (Standard) or B (Factory).
        # Based on the function signature of pad_object (returns Tensor) and the provided paths,
        # this is Scenario A.
        
        outer_path = None
        inner_paths = []

        for p in data_paths:
            if 'standard_data_pad_object.pkl' in p:
                outer_path = p
            elif 'standard_data_parent_function_pad_object_' in p:
                inner_paths.append(p)
        
        if not outer_path:
            print("Error: standard_data_pad_object.pkl not found in provided paths.")
            sys.exit(1)

        print(f"Loading primary data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        outer_expected = outer_data.get('output', None)

        # 2. EXECUTION
        print("Executing pad_object with loaded arguments...")
        try:
            # Run the function
            result = pad_object(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"Execution Error during pad_object call: {e}")
            traceback.print_exc()
            sys.exit(1)

        # 3. VERIFICATION STRATEGY
        # Check if the result is a callable (Closure/Factory pattern) or a value
        if callable(result) and inner_paths:
            print("Detected Closure/Factory pattern (Scenario B). Testing inner function calls...")
            
            # Scenario B: The result is an operator, we must test it against inner data files
            operator = result
            all_passed = True
            
            for inner_path in inner_paths:
                print(f"  Testing against inner data: {os.path.basename(inner_path)}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output', None)
                
                try:
                    inner_result = operator(*inner_args, **inner_kwargs)
                except Exception as e:
                    print(f"  Execution Error during inner call: {e}")
                    traceback.print_exc()
                    all_passed = False
                    continue
                
                passed, msg = recursive_check(inner_expected, inner_result)
                if not passed:
                    print(f"  FAILED: {msg}")
                    all_passed = False
                else:
                    print(f"  Passed.")

            if all_passed:
                print("TEST PASSED")
                sys.exit(0)
            else:
                print("TEST FAILED: One or more inner checks failed.")
                sys.exit(1)

        else:
            # Scenario A: The result is the final value (Tensor)
            print("Detected Standard Execution pattern (Scenario A). Verifying output...")
            
            passed, msg = recursive_check(outer_expected, result)
            
            if passed:
                print("TEST PASSED")
                sys.exit(0)
            else:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)

    except Exception as e:
        print(f"An unexpected error occurred in test wrapper: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()