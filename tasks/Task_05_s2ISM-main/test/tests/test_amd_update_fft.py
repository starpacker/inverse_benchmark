import sys
import os
import dill
import torch
import numpy as np
import traceback
import string

# Ensure the target module is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the target function
try:
    from agent_amd_update_fft import amd_update_fft
except ImportError:
    print("Error: Could not import 'amd_update_fft' from 'agent_amd_update_fft.py'.")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    # Define a fallback if verification_utils is missing
    def recursive_check(expected, actual):
        if isinstance(expected, torch.Tensor):
            if not isinstance(actual, torch.Tensor):
                return False, f"Expected torch.Tensor, got {type(actual)}"
            if expected.shape != actual.shape:
                return False, f"Shape mismatch: expected {expected.shape}, got {actual.shape}"
            # Check for NaN/Inf before comparison
            if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
                return False, "Tensors contain NaN or Inf values."
            if not torch.allclose(expected, actual, rtol=1e-3, atol=1e-4):
                return False, f"Tensor values mismatch. Max diff: {(expected - actual).abs().max()}"
            return True, "Success"
        elif isinstance(expected, (list, tuple)):
            if len(expected) != len(actual):
                return False, "Length mismatch"
            for e, a in zip(expected, actual):
                res, msg = recursive_check(e, a)
                if not res:
                    return False, msg
            return True, "Success"
        else:
            if expected != actual:
                return False, f"Value mismatch: {expected} != {actual}"
            return True, "Success"

def test_amd_update_fft():
    # Data paths provided
    data_paths = ['/data/yjh/s2ISM-main_sandbox/run_code/std_data/standard_data_amd_update_fft.pkl']
    
    # Identify the main data file
    target_data_path = None
    for path in data_paths:
        if 'standard_data_amd_update_fft.pkl' in path and 'parent_function' not in path:
            target_data_path = path
            break

    if not target_data_path or not os.path.exists(target_data_path):
        print(f"Error: Data file not found at {target_data_path}")
        sys.exit(1)

    # Load input data
    try:
        with open(target_data_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Error loading data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_output = data.get('output', None)

    print(f"Loaded test data for function: {data.get('func_name')}")

    # Move tensor inputs to GPU if available and if the original inputs were on GPU
    # Since we can't easily know where they were, we usually respect the environment.
    # However, the provided function has a specific `torch.cuda.empty_cache()` call,
    # hinting it expects GPU usage.
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def move_to_device(obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, (list, tuple)):
            return type(obj)(move_to_device(x, device) for x in obj)
        elif isinstance(obj, dict):
            return {k: move_to_device(v, device) for k, v in obj.items()}
        return obj

    args = move_to_device(args, device)
    kwargs = move_to_device(kwargs, device)

    # Execute the function
    try:
        print("Executing amd_update_fft...")
        actual_output = amd_update_fft(*args, **kwargs)
    except Exception as e:
        print(f"Error executing amd_update_fft: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Move expected output to same device for comparison
    expected_output = move_to_device(expected_output, device)

    # Verify results
    try:
        passed, msg = recursive_check(expected_output, actual_output)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    except Exception as e:
        print(f"Error during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_amd_update_fft()