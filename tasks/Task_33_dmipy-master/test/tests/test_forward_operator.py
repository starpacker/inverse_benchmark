import sys
import os
import dill
import numpy as np
import traceback

# Defensive import for torch to prevent ModuleNotFoundError
try:
    import torch
except ImportError:
    torch = None

from agent_forward_operator import forward_operator

# -------------------------------------------------------------------------
# Verification Utility
# -------------------------------------------------------------------------
def recursive_check(expected, actual, rtol=1e-5, atol=1e-8):
    """
    Recursively compares expected and actual structures.
    Supports dicts, lists, tuples, numpy arrays, and torch tensors.
    """
    if expected is None:
        return actual is None, f"Expected None, got {type(actual)}"
    
    # Handle simple types
    if isinstance(expected, (int, float, str, bool)):
        if expected != actual:
            # Allow slight float differences for scalars
            if isinstance(expected, (float, int)) and isinstance(actual, (float, int)):
                if not np.isclose(expected, actual, rtol=rtol, atol=atol):
                    return False, f"Scalar mismatch: expected {expected}, got {actual}"
                return True, ""
            return False, f"Value mismatch: expected {expected}, got {actual}"
        return True, ""

    # Handle Numpy Arrays
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray):
            return False, f"Type mismatch: expected np.ndarray, got {type(actual)}"
        if expected.shape != actual.shape:
            return False, f"Shape mismatch: expected {expected.shape}, got {actual.shape}"
        if not np.allclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True):
            diff = np.abs(expected - actual)
            return False, f"Array mismatch. Max diff: {np.max(diff)}"
        return True, ""

    # Handle Torch Tensors
    if torch and isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            return False, f"Type mismatch: expected torch.Tensor, got {type(actual)}"
        # Move to CPU and numpy for comparison
        exp_np = expected.detach().cpu().numpy()
        act_np = actual.detach().cpu().numpy()
        if exp_np.shape != act_np.shape:
            return False, f"Tensor shape mismatch: expected {exp_np.shape}, got {act_np.shape}"
        if not np.allclose(exp_np, act_np, rtol=rtol, atol=atol, equal_nan=True):
            diff = np.abs(exp_np - act_np)
            return False, f"Tensor mismatch. Max diff: {np.max(diff)}"
        return True, ""

    # Handle Lists/Tuples
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)):
            return False, f"Type mismatch: expected {type(expected)}, got {type(actual)}"
        if len(expected) != len(actual):
            return False, f"Length mismatch: expected {len(expected)}, got {len(actual)}"
        for i, (e_item, a_item) in enumerate(zip(expected, actual)):
            passed, msg = recursive_check(e_item, a_item, rtol, atol)
            if not passed:
                return False, f"Index {i}: {msg}"
        return True, ""

    # Handle Dictionaries
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False, f"Type mismatch: expected dict, got {type(actual)}"
        if set(expected.keys()) != set(actual.keys()):
            return False, f"Key mismatch: expected {list(expected.keys())}, got {list(actual.keys())}"
        for k, v in expected.items():
            passed, msg = recursive_check(v, actual[k], rtol, atol)
            if not passed:
                return False, f"Key '{k}': {msg}"
        return True, ""

    # Fallback
    if expected != actual:
        return False, f"Generic mismatch: expected {expected}, got {actual}"
    
    return True, ""

# -------------------------------------------------------------------------
# Test Script
# -------------------------------------------------------------------------
def test_forward_operator():
    # Define paths
    base_data_path = '/data/yjh/dmipy-master_sandbox/run_code/std_data/standard_data_forward_operator.pkl'
    
    # 1. Load Data
    if not os.path.exists(base_data_path):
        print(f"Error: Data file not found at {base_data_path}")
        sys.exit(1)
        
    try:
        with open(base_data_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_output = data.get('output', None)

    # 2. Execution (Scenario A: Direct Function)
    # The provided code for forward_operator returns an array, not a callable, 
    # so we treat it as a direct function call test.
    print(f"Executing forward_operator with {len(args)} args and {len(kwargs)} kwargs...")
    
    try:
        actual_result = forward_operator(*args, **kwargs)
    except Exception as e:
        print(f"Error executing forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Verification
    print("Verifying results...")
    passed, msg = recursive_check(expected_output, actual_result)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    test_forward_operator()