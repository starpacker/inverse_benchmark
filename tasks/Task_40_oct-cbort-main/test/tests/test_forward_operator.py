import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add current directory to path so we can import the function
sys.path.append(os.getcwd())

# Import the target function
try:
    from agent_forward_operator import forward_operator
except ImportError:
    print("Error: Could not import 'forward_operator' from 'agent_forward_operator.py'")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    # Fallback if verification_utils not found, though instructions say it's available
    def recursive_check(expected, actual, rtol=1e-4, atol=1e-5):
        if isinstance(expected, (list, tuple)):
            if len(expected) != len(actual):
                return False, f"Length mismatch: {len(expected)} vs {len(actual)}"
            for i, (e, a) in enumerate(zip(expected, actual)):
                ok, msg = recursive_check(e, a, rtol, atol)
                if not ok: return False, f"Index {i}: {msg}"
            return True, "OK"
        if isinstance(expected, np.ndarray):
            if not np.allclose(expected, actual, rtol=rtol, atol=atol):
                diff = np.abs(expected - actual)
                return False, f"Numpy mismatch. Max diff: {np.max(diff)}"
            return True, "OK"
        if isinstance(expected, torch.Tensor):
            if not torch.allclose(expected.cpu(), actual.cpu(), rtol=rtol, atol=atol):
                diff = (expected - actual).abs()
                return False, f"Torch mismatch. Max diff: {diff.max().item()}"
            return True, "OK"
        if expected != actual:
            return False, f"Value mismatch: {expected} != {actual}"
        return True, "OK"

def main():
    # 1. Define Data Paths
    data_paths = ['/data/yjh/oct-cbort-main_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # 2. Identify Test Scenario
    # We look for a standard data file that corresponds exactly to the function name,
    # and potentially files that correspond to a closure (parent function pattern).
    outer_path = None
    inner_paths = []

    for path in data_paths:
        if 'standard_data_forward_operator.pkl' in path:
            outer_path = path
        elif 'standard_data_parent_function_forward_operator' in path:
            inner_paths.append(path)
    
    if not outer_path:
        print("Test Skipped: No standard_data_forward_operator.pkl found.")
        sys.exit(0)

    print(f"Loading data from {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Execution Phase
    try:
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)

        # Execute the function
        actual_output = forward_operator(*args, **kwargs)
        
        # Check if the result is a callable (Closure/Factory Pattern)
        # However, looking at the provided source code for `forward_operator`, it returns (tom1, tom2) directly.
        # It is NOT a factory function. It performs computation immediately.
        # But we must respect the potential for the Data Capture Decorator to have captured a closure structure if it existed.
        # Based on the user-provided code in the prompt, forward_operator returns a tuple of arrays.
        
        # If inner_paths existed, it would imply a closure, but the provided code shows a direct calculation.
        # We proceed with direct comparison.
        
        # 4. Verification
        passed, msg = recursive_check(expected_output, actual_output)

        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            # Debugging info
            if isinstance(actual_output, (tuple, list)) and len(actual_output) > 0:
                 if hasattr(actual_output[0], 'shape'):
                     print(f"Actual shape[0]: {actual_output[0].shape}")
            if isinstance(expected_output, (tuple, list)) and len(expected_output) > 0:
                 if hasattr(expected_output[0], 'shape'):
                     print(f"Expected shape[0]: {expected_output[0].shape}")
            sys.exit(1)

    except Exception as e:
        print(f"Runtime Error during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()