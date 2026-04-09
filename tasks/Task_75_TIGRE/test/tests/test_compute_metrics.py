import sys
import os
import dill
import numpy as np
import traceback
import math

# Import the target function
from agent_compute_metrics import compute_metrics

# Import verification utility
from verification_utils import recursive_check

def custom_recursive_check(expected, actual, path="output"):
    """
    Custom recursive check that handles NaN values properly.
    NaN == NaN should be considered True for test purposes.
    """
    # Handle NaN cases for floats
    if isinstance(expected, float) and isinstance(actual, float):
        if math.isnan(expected) and math.isnan(actual):
            return True, "Both are NaN (considered equal)"
        if math.isnan(expected) or math.isnan(actual):
            return False, f"Float mismatch at {path}: expected {expected}, got {actual}"
        # For regular floats, use relative tolerance
        if abs(expected) < 1e-12:
            if abs(actual - expected) > 1e-9:
                return False, f"Float mismatch at {path}: expected {expected}, got {actual}"
        else:
            rel_diff = abs(actual - expected) / abs(expected)
            if rel_diff > 1e-6:
                return False, f"Float mismatch at {path}: expected {expected}, got {actual}, rel_diff={rel_diff}"
        return True, "Floats match"
    
    # Handle numpy arrays
    if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
        if expected.shape != actual.shape:
            return False, f"Shape mismatch at {path}: expected {expected.shape}, got {actual.shape}"
        if expected.dtype != actual.dtype:
            # Allow dtype differences if values are close
            pass
        # Check for NaN positions
        expected_nan = np.isnan(expected)
        actual_nan = np.isnan(actual)
        if not np.array_equal(expected_nan, actual_nan):
            return False, f"NaN positions mismatch at {path}"
        # Compare non-NaN values
        mask = ~expected_nan
        if np.any(mask):
            if not np.allclose(expected[mask], actual[mask], rtol=1e-6, atol=1e-9):
                return False, f"Array values mismatch at {path}"
        return True, "Arrays match"
    
    # Handle dictionaries
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected.keys()) != set(actual.keys()):
            return False, f"Dict keys mismatch at {path}: expected {set(expected.keys())}, got {set(actual.keys())}"
        for key in expected.keys():
            passed, msg = custom_recursive_check(expected[key], actual[key], f"{path}['{key}']")
            if not passed:
                return False, msg
        return True, "Dicts match"
    
    # Handle lists/tuples
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return False, f"Length mismatch at {path}: expected {len(expected)}, got {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            passed, msg = custom_recursive_check(e, a, f"{path}[{i}]")
            if not passed:
                return False, msg
        return True, "Sequences match"
    
    # Handle None
    if expected is None and actual is None:
        return True, "Both are None"
    
    # Handle other types
    if type(expected) != type(actual):
        # Allow int/float comparison
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return custom_recursive_check(float(expected), float(actual), path)
        return False, f"Type mismatch at {path}: expected {type(expected)}, got {type(actual)}"
    
    if expected != actual:
        return False, f"Value mismatch at {path}: expected {expected}, got {actual}"
    
    return True, "Values match"


def main():
    # Data paths provided
    data_paths = ['/data/yjh/TIGRE_sandbox_sandbox/run_code/std_data/standard_data_compute_metrics.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_compute_metrics.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_compute_metrics.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    try:
        result = compute_metrics(*outer_args, **outer_kwargs)
        print("Successfully executed compute_metrics with outer data")
    except Exception as e:
        print(f"ERROR: Failed to execute compute_metrics: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")
        
        if not callable(result):
            print("ERROR: Expected callable operator from outer function, but got non-callable")
            sys.exit(1)
        
        agent_operator = result
        
        # Process each inner path
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed agent_operator with inner data")
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results using custom check that handles NaN
            passed, msg = custom_recursive_check(expected, actual_result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("Detected Scenario A: Simple function")
        expected = outer_output
        
        # Verify results using custom check that handles NaN
        passed, msg = custom_recursive_check(expected, result)
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()