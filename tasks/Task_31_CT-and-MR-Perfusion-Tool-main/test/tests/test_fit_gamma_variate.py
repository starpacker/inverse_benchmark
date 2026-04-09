import sys
import os
import dill
import numpy as np
import traceback
import warnings

# Attempt to import torch gracefully, as the environment might lack it
try:
    import torch
except ImportError:
    torch = None

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure local modules can be imported
sys.path.append(os.getcwd())

# Import the target function
try:
    from agent_fit_gamma_variate import fit_gamma_variate
except ImportError:
    print("CRITICAL: Could not import 'fit_gamma_variate' from 'agent_fit_gamma_variate'.")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    # Robust fallback if verification_utils is missing
    def recursive_check(expected, actual):
        if expected is None:
            return (actual is None), f"Expected None, got {type(actual)}"
        if isinstance(expected, (list, tuple)):
            if len(expected) != len(actual):
                return False, f"Length mismatch: {len(expected)} vs {len(actual)}"
            for i, (e, a) in enumerate(zip(expected, actual)):
                res, msg = recursive_check(e, a)
                if not res: return False, f"Index {i}: {msg}"
            return True, ""
        if isinstance(expected, np.ndarray):
            if not isinstance(actual, np.ndarray):
                return False, f"Type mismatch: expected ndarray, got {type(actual)}"
            # Allow small float differences
            if expected.shape != actual.shape:
                return False, f"Shape mismatch: {expected.shape} vs {actual.shape}"
            # Handle NaNs and Infs equal
            if not np.allclose(expected, actual, equal_nan=True, atol=1e-4):
                return False, f"Array mismatch. Max diff: {np.max(np.abs(expected - actual))}"
            return True, ""
        if expected != actual:
            return False, f"Value mismatch: {expected} != {actual}"
        return True, ""

def test_fit_gamma_variate():
    # 1. SETUP DATA PATHS
    # The prompt provides specific paths. We target the main data file.
    data_paths = ['/data/yjh/CT-and-MR-Perfusion-Tool-main_sandbox/run_code/std_data/standard_data_fit_gamma_variate.pkl']
    
    outer_path = None
    for p in data_paths:
        if 'standard_data_fit_gamma_variate.pkl' in p:
            outer_path = p
            break
            
    if not outer_path or not os.path.exists(outer_path):
        print(f"TEST FAILED: Data file not found at {outer_path}")
        sys.exit(1)

    # 2. LOAD DATA
    try:
        with open(outer_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"TEST FAILED: Error loading pickle data: {e}")
        traceback.print_exc()
        sys.exit(1)

    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_output = data.get('output')

    # 3. EXECUTE TARGET FUNCTION
    # Based on code analysis, fit_gamma_variate is a simple function returning data (Scenario A),
    # not a factory returning a closure.
    print(f"Executing fit_gamma_variate with {len(args)} args and {len(kwargs)} kwargs...")
    
    try:
        actual_output = fit_gamma_variate(*args, **kwargs)
    except Exception as e:
        print(f"TEST FAILED: Execution crashed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. VERIFY RESULTS
    passed, msg = recursive_check(expected_output, actual_output)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        # Debug info
        print("-" * 30)
        print("Expected Output Sample:", expected_output)
        print("Actual Output Sample:  ", actual_output)
        print("-" * 30)
        sys.exit(1)

if __name__ == "__main__":
    test_fit_gamma_variate()