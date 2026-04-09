import sys
import os
import dill
import numpy as np
import traceback

# Handle torch soft import to prevent ModuleNotFoundError if not installed
try:
    import torch
except ImportError:
    torch = None

# Add current directory to sys.path to ensure local imports work
sys.path.append(os.getcwd())

# Import the target function
try:
    from agent_parse_harmonic import parse_harmonic
except ImportError as e:
    print(f"CRITICAL: Could not import 'parse_harmonic' from 'agent_parse_harmonic'. Error: {e}")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    # Fallback if verification_utils is missing, though it's required by instruction
    print("WARNING: verification_utils not found. Defining fallback recursive_check.")
    def recursive_check(expected, actual):
        if expected == actual:
            return True, "Match"
        try:
            # Handle numpy arrays
            if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
                if np.array_equal(expected, actual):
                    return True, "Match (numpy)"
                if np.allclose(expected, actual):
                    return True, "Match (numpy close)"
            # Handle lists/tuples recursively
            if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
                if len(expected) != len(actual):
                    return False, f"Length mismatch: {len(expected)} vs {len(actual)}"
                for i, (e, a) in enumerate(zip(expected, actual)):
                    res, msg = recursive_check(e, a)
                    if not res:
                        return False, f"Index {i}: {msg}"
                return True, "Match (sequence)"
        except Exception as e:
            pass
        return False, f"Expected {expected}, got {actual}"

def load_data(path):
    """Safely load data using dill."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return dill.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def run_test():
    # 1. SETUP PATHS
    data_paths = ['/data/yjh/phasorpy-main_sandbox/run_code/std_data/standard_data_parse_harmonic.pkl']
    
    # Identify relevant data file
    outer_path = None
    for p in data_paths:
        if 'standard_data_parse_harmonic.pkl' in p:
            outer_path = p
            break
            
    if not outer_path:
        print("Test skipped: No standard_data_parse_harmonic.pkl found.")
        sys.exit(0)

    # 2. LOAD DATA
    print(f"Loading data from {outer_path}...")
    outer_data = load_data(outer_path)
    if outer_data is None:
        print("CRITICAL: Failed to load data file.")
        sys.exit(1)

    args = outer_data.get('args', [])
    kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    print(f"Inputs - Args: {args}, Kwargs: {kwargs}")

    # 3. EXECUTE TARGET FUNCTION
    try:
        # Scenario A: Simple Function Execution
        # Based on function signature, parse_harmonic returns (list, bool), it is not a factory.
        actual_result = parse_harmonic(*args, **kwargs)
    except Exception as e:
        print(f"Execution Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. VERIFY
    print("Verifying results...")
    is_match, msg = recursive_check(expected_output, actual_result)

    if is_match:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        print(f"Expected: {expected_output}")
        print(f"Actual:   {actual_result}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()