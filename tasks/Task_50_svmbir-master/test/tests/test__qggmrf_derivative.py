import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the directory containing the target function to the python path
sys.path.append(os.path.dirname(__file__))

# Import the target function
try:
    from agent__qggmrf_derivative import _qggmrf_derivative
except ImportError:
    print("Error: Could not import _qggmrf_derivative from agent__qggmrf_derivative.py")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    # If verification_utils is not available, define a simple fallback
    def recursive_check(expected, actual):
        if isinstance(expected, np.ndarray):
            if not np.allclose(expected, actual, equal_nan=True):
                return False, f"Numpy arrays do not match. Expected shape {expected.shape}, actual {actual.shape}"
            return True, ""
        elif hasattr(expected, 'cpu') and hasattr(actual, 'cpu'): # Torch tensors
             if not torch.allclose(expected.cpu(), actual.cpu(), equal_nan=True):
                 return False, "Torch tensors do not match"
             return True, ""
        elif expected != actual:
            return False, f"Values mismatch: {expected} vs {actual}"
        return True, ""

def test_qggmrf_derivative():
    data_paths = ['/data/yjh/svmbir-master_sandbox/run_code/std_data/standard_data__qggmrf_derivative.pkl']
    
    # 1. Identify File Types
    outer_data_path = None
    inner_data_path = None

    for path in data_paths:
        if 'parent_function' in path:
            inner_data_path = path
        elif 'standard_data__qggmrf_derivative.pkl' in path:
            outer_data_path = path

    if not outer_data_path:
        print("Error: standard_data__qggmrf_derivative.pkl not found in provided paths.")
        sys.exit(1)

    # 2. Load Outer Data (Main Function Arguments)
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"Loaded outer data from {outer_data_path}")
    except Exception as e:
        print(f"Error loading outer data: {e}")
        sys.exit(1)

    # 3. Execute Target Function
    try:
        print("Executing _qggmrf_derivative with loaded arguments...")
        # Unpack arguments
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        
        # Run function
        actual_result = _qggmrf_derivative(*args, **kwargs)
        
    except Exception as e:
        print("Error executing _qggmrf_derivative:")
        traceback.print_exc()
        sys.exit(1)

    # 4. Handle Return Type (Closure vs Direct Result)
    # Based on the function definition provided in the prompt, _qggmrf_derivative returns a value (grad/sigma^p), not a function.
    # However, the prompt mentions checking for Factory/Closure pattern.
    # Given only one data file exists and the code computes a gradient directly, it's likely a direct calculation.
    
    expected_result = None
    
    if inner_data_path and callable(actual_result):
        # Scenario: Factory Pattern where function returns a closure
        try:
            with open(inner_data_path, 'rb') as f:
                inner_data = dill.load(f)
            print(f"Loaded inner data from {inner_data_path}")
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_result = inner_data.get('output')
            
            print("Executing returned closure with inner arguments...")
            actual_result = actual_result(*inner_args, **inner_kwargs)
            
        except Exception as e:
            print(f"Error executing closure or loading inner data: {e}")
            sys.exit(1)
    else:
        # Scenario: Direct Calculation
        expected_result = outer_data.get('output')

    # 5. Verification
    try:
        passed, msg = recursive_check(expected_result, actual_result)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            # Debugging info
            if isinstance(expected_result, (np.ndarray, torch.Tensor)):
                 print(f"Expected shape: {expected_result.shape}")
                 print(f"Actual shape: {actual_result.shape}")
                 print(f"Expected mean: {np.mean(expected_result) if isinstance(expected_result, np.ndarray) else expected_result.float().mean()}")
                 print(f"Actual mean: {np.mean(actual_result) if isinstance(actual_result, np.ndarray) else actual_result.float().mean()}")
            sys.exit(1)
    except Exception as e:
        print(f"Error during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_qggmrf_derivative()