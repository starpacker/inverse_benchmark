import sys
import os
import dill
import numpy as np
import traceback

# Handle missing torch gracefully to prevent ModuleNotFoundError
try:
    import torch
except ImportError:
    torch = None

# Ensure the current directory is in sys.path to import local modules
sys.path.append(os.getcwd())

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

def run_test():
    # Define the data paths as provided
    data_paths = ['/data/yjh/storm-analysis-master_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Identify the correct data file for the function
    pkl_path = None
    for p in data_paths:
        if 'standard_data_load_and_preprocess_data.pkl' in p:
            pkl_path = p
            break
            
    if not pkl_path:
        print("Error: Standard data file path not found in provided list.")
        sys.exit(1)

    if not os.path.exists(pkl_path):
        print(f"Error: Data file does not exist at {pkl_path}")
        sys.exit(1)

    # Load the captured data
    try:
        with open(pkl_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle data: {e}")
        # If unpickling fails specifically because of missing torch reference in pickle, try mocking
        if "torch" in str(e) and torch is None:
            print("Attempting to mock torch for unpickling...")
            from unittest.mock import MagicMock
            sys.modules['torch'] = MagicMock()
            try:
                with open(pkl_path, 'rb') as f:
                    data = dill.load(f)
            except Exception as e2:
                print(f"Retry failed: {e2}")
                sys.exit(1)
        else:
            traceback.print_exc()
            sys.exit(1)

    # Extract arguments and expected output
    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_output = data.get('output')

    # Pre-flight check: The function relies on an external file path (args[0])
    # If that file doesn't exist, the test will crash. We check and warn.
    if args and len(args) > 0 and isinstance(args[0], str):
        target_file_path = args[0]
        if not os.path.exists(target_file_path):
            print(f"WARNING: The file path argument '{target_file_path}' does not exist on this system.")
            print("The test is expected to fail with FileNotFoundError unless the environment is identical to the capture environment.")

    # Execute the function
    try:
        # Scenario A: Simple function execution
        result = load_and_preprocess_data(*args, **kwargs)
    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Verify the result
    try:
        passed, msg = recursive_check(expected_output, result)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    except Exception as e:
        print(f"Verification failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()