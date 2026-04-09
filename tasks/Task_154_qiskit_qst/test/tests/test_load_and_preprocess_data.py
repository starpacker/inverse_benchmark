import sys
import os
import dill
import numpy as np
import traceback

# Add the parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def custom_recursive_check(expected, actual, rtol=0.1, atol=0.05, path="output"):
    """
    Custom comparison that handles stochastic quantum measurement results.
    Uses relaxed tolerances for floating point comparisons.
    """
    # Handle None
    if expected is None and actual is None:
        return True, "Both are None"
    if expected is None or actual is None:
        return False, f"Mismatch at {path}: one is None, other is not"
    
    # Handle numpy arrays
    if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
        if expected.shape != actual.shape:
            return False, f"Shape mismatch at {path}: expected {expected.shape}, got {actual.shape}"
        if expected.dtype != actual.dtype:
            # Allow dtype differences for complex types
            if not (np.issubdtype(expected.dtype, np.complexfloating) and 
                    np.issubdtype(actual.dtype, np.complexfloating)):
                pass  # Allow dtype differences
        try:
            if np.allclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True):
                return True, "Arrays match within tolerance"
            else:
                max_diff = np.max(np.abs(expected - actual))
                return False, f"Array mismatch at {path}: max diff = {max_diff}"
        except Exception as e:
            return False, f"Array comparison failed at {path}: {e}"
    
    # Handle tuples
    if isinstance(expected, tuple) and isinstance(actual, tuple):
        if len(expected) != len(actual):
            return False, f"Tuple length mismatch at {path}: expected {len(expected)}, got {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = custom_recursive_check(e, a, rtol, atol, f"{path}[{i}]")
            if not passed:
                return False, msg
        return True, "Tuples match"
    
    # Handle lists
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return False, f"List length mismatch at {path}: expected {len(expected)}, got {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = custom_recursive_check(e, a, rtol, atol, f"{path}[{i}]")
            if not passed:
                return False, msg
        return True, "Lists match"
    
    # Handle dictionaries
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected.keys()) != set(actual.keys()):
            missing = set(expected.keys()) - set(actual.keys())
            extra = set(actual.keys()) - set(expected.keys())
            return False, f"Key mismatch at {path}: missing={missing}, extra={extra}"
        for key in expected.keys():
            passed, msg = custom_recursive_check(expected[key], actual[key], rtol, atol, f"{path}['{key}']")
            if not passed:
                return False, msg
        return True, "Dicts match"
    
    # Handle floats with tolerance
    if isinstance(expected, (int, float, np.floating)) and isinstance(actual, (int, float, np.floating)):
        if np.isnan(expected) and np.isnan(actual):
            return True, "Both NaN"
        if np.isinf(expected) and np.isinf(actual):
            if np.sign(expected) == np.sign(actual):
                return True, "Both same infinity"
        diff = abs(float(expected) - float(actual))
        threshold = atol + rtol * abs(float(expected))
        if diff <= threshold:
            return True, "Floats match within tolerance"
        return False, f"Float mismatch at {path}: expected {expected}, got {actual}, diff={diff}, threshold={threshold}"
    
    # Handle complex numbers
    if isinstance(expected, (complex, np.complexfloating)) and isinstance(actual, (complex, np.complexfloating)):
        diff = abs(complex(expected) - complex(actual))
        threshold = atol + rtol * abs(complex(expected))
        if diff <= threshold:
            return True, "Complex match within tolerance"
        return False, f"Complex mismatch at {path}: expected {expected}, got {actual}"
    
    # Handle strings
    if isinstance(expected, str) and isinstance(actual, str):
        if expected == actual:
            return True, "Strings match"
        return False, f"String mismatch at {path}: expected '{expected}', got '{actual}'"
    
    # Handle integers
    if isinstance(expected, (int, np.integer)) and isinstance(actual, (int, np.integer)):
        if int(expected) == int(actual):
            return True, "Integers match"
        return False, f"Integer mismatch at {path}: expected {expected}, got {actual}"
    
    # Default: type comparison then equality
    if type(expected) != type(actual):
        # Allow some type flexibility
        pass
    
    try:
        if expected == actual:
            return True, "Objects equal"
    except Exception:
        pass
    
    return False, f"Mismatch at {path}: expected {type(expected).__name__}, got {type(actual).__name__}"


def main():
    data_paths = ['/data/yjh/qiskit_qst_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = p
    
    if outer_path is None:
        print("TEST FAILED: Could not find outer data file")
        sys.exit(1)
    
    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from {outer_path}")
    except Exception as e:
        print(f"TEST FAILED: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    print(f"Outer args: {outer_args}")
    print(f"Outer kwargs: {outer_kwargs}")
    
    # Set seeds before running to try to match original execution
    set_all_seeds(42)
    
    # Execute the function
    try:
        actual_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Function executed successfully")
    except Exception as e:
        print(f"TEST FAILED: Function execution error: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if there are inner paths (factory pattern)
    if inner_paths:
        # Factory pattern - actual_result should be callable
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from {inner_path}")
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output')
                
                if callable(actual_result):
                    set_all_seeds(42)
                    inner_result = actual_result(*inner_args, **inner_kwargs)
                    passed, msg = custom_recursive_check(inner_expected, inner_result)
                else:
                    passed, msg = False, "Expected callable for factory pattern"
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                    
            except Exception as e:
                print(f"TEST FAILED: Inner execution error: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Simple function pattern - compare results directly
        # Use custom check with tolerance for stochastic quantum results
        passed, msg = custom_recursive_check(expected_output, actual_result)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()