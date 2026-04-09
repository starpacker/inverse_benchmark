import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add current directory to path so we can import the target module
sys.path.append(os.path.dirname(__file__))

# Import the target function
try:
    from agent_pad2d import pad2d
except ImportError:
    print("Error: Could not import 'pad2d' from 'agent_pad2d.py'")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    # Fallback if verification_utils is missing in local context (for standalone safety)
    def recursive_check(expected, actual):
        if isinstance(expected, np.ndarray):
            if not isinstance(actual, np.ndarray):
                return False, f"Expected numpy array, got {type(actual)}"
            if expected.shape != actual.shape:
                return False, f"Shape mismatch: expected {expected.shape}, got {actual.shape}"
            if not np.allclose(expected, actual, equal_nan=True):
                return False, "Values mismatch"
            return True, ""
        if isinstance(expected, torch.Tensor):
            if not isinstance(actual, torch.Tensor):
                return False, f"Expected tensor, got {type(actual)}"
            if expected.shape != actual.shape:
                return False, f"Shape mismatch: expected {expected.shape}, got {actual.shape}"
            if not torch.allclose(expected, actual, equal_nan=True):
                return False, "Values mismatch"
            return True, ""
        if expected != actual:
            return False, f"Expected {expected}, got {actual}"
        return True, ""

def main():
    data_paths = ['/data/yjh/mripy-master_sandbox/run_code/std_data/standard_data_pad2d.pkl']
    
    # 1. Identify File Roles
    # In this specific case, pad2d is a standard function, not a factory, based on the signature provided.
    # So we look for the main data file.
    primary_data_path = None
    
    for path in data_paths:
        if 'standard_data_pad2d.pkl' in path:
            primary_data_path = path
            break
            
    if not primary_data_path or not os.path.exists(primary_data_path):
        print(f"Error: Data file not found at {primary_data_path}")
        sys.exit(1)

    # 2. Load Data
    try:
        with open(primary_data_path, 'rb') as f:
            data_payload = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    func_name = data_payload.get('func_name')
    args = data_payload.get('args', [])
    kwargs = data_payload.get('kwargs', {})
    expected_output = data_payload.get('output')

    print(f"Loaded data for function: {func_name}")
    print(f"Args type: {[type(a) for a in args]}")

    # 3. Execution
    try:
        # Run the target function
        actual_output = pad2d(*args, **kwargs)
    except Exception as e:
        print(f"Error executing pad2d: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verification
    passed, msg = recursive_check(expected_output, actual_output)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        # Detailed debug info for arrays
        if isinstance(expected_output, np.ndarray) and isinstance(actual_output, np.ndarray):
            print(f"Expected shape: {expected_output.shape}, Actual shape: {actual_output.shape}")
            print(f"Max diff: {np.max(np.abs(expected_output - actual_output))}")
        sys.exit(1)

if __name__ == "__main__":
    main()