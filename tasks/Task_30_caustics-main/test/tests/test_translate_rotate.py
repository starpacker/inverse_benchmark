import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
try:
    from agent_translate_rotate import translate_rotate
except ImportError:
    print("Error: Could not import 'translate_rotate' from 'agent_translate_rotate'. Check your python path.")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    print("Error: Could not import 'recursive_check' from 'verification_utils'.")
    sys.exit(1)

def test_translate_rotate():
    # 1. DATA FILE SETUP
    # The paths provided in the instructions
    data_paths = ['/data/yjh/caustics-main_sandbox/run_code/std_data/standard_data_translate_rotate.pkl']
    
    # Identify the main data file
    target_path = None
    for path in data_paths:
        if 'standard_data_translate_rotate.pkl' in path:
            target_path = path
            break
            
    if not target_path or not os.path.exists(target_path):
        print(f"Error: Data file not found at {target_path}")
        sys.exit(1)

    print(f"Loading data from: {target_path}")

    # 2. LOAD DATA
    try:
        with open(target_path, 'rb') as f:
            data_payload = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Extract inputs and expected output
    func_args = data_payload.get('args', [])
    func_kwargs = data_payload.get('kwargs', {})
    expected_output = data_payload.get('output', None)

    # 3. EXECUTE TARGET FUNCTION
    # translate_rotate returns a tuple of Tensors (xt, yt), it is not a factory/closure.
    # Therefore, we execute it directly and compare the results.
    print(f"Executing translate_rotate with {len(func_args)} args and {len(func_kwargs)} kwargs...")
    
    try:
        actual_output = translate_rotate(*func_args, **func_kwargs)
    except Exception as e:
        print(f"Error during function execution: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. VERIFICATION
    # Check if the result is callable (Scenario B detection). 
    # For translate_rotate, we expect a tuple of tensors, not a callable.
    if callable(actual_output):
        print("Unexpected: translate_rotate returned a callable, but was expected to return data (Scenario A).")
        # If it were a factory, we would look for child data files here, 
        # but analysis of the function code confirms it returns values.
        sys.exit(1)

    print("Verifying results...")
    try:
        passed, msg = recursive_check(expected_output, actual_output)
    except Exception as e:
        print(f"Error during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    test_translate_rotate()