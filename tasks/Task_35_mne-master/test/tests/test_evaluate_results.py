import sys
import os
import dill
import numpy as np
import traceback

# Add the current directory to path to ensure imports work
sys.path.append(os.getcwd())

# Import the target function
try:
    from agent_evaluate_results import evaluate_results
except ImportError:
    print("Error: Could not import 'evaluate_results' from 'agent_evaluate_results.py'.")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    # Fallback if verification_utils is missing, though instructions say it exists
    def recursive_check(expected, actual):
        if isinstance(expected, np.ndarray):
            if not np.allclose(expected, actual, rtol=1e-5, atol=1e-8):
                return False, f"Numpy arrays differ. Expected shape {expected.shape}, got {actual.shape}"
            return True, ""
        if expected != actual:
            return False, f"Values differ: Expected {expected} vs Actual {actual}"
        return True, ""

def main():
    data_paths = ['/data/yjh/mne-master_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # 1. Strategy Determination
    outer_path = None
    inner_paths = []

    for path in data_paths:
        if 'standard_data_evaluate_results.pkl' in path:
            outer_path = path
        elif 'standard_data_parent_function_evaluate_results' in path:
            inner_paths.append(path)
    
    if not outer_path:
        print("Error: No standard_data_evaluate_results.pkl found in provided paths.")
        sys.exit(1)

    print(f"Loading Outer Data: {outer_path}")
    
    # 2. Load Outer Data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer data: {e}")
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_outer_output = outer_data.get('output', None)

    # 3. Phase 1: Run the function (Create Operator or Get Result)
    print("Running evaluate_results...")
    try:
        # evaluate_results in this context usually returns None (it prints/plots) 
        # or might return metrics. We execute it to ensure it runs without crashing.
        actual_result = evaluate_results(*outer_args, **outer_kwargs)
    except Exception:
        traceback.print_exc()
        print("Execution of evaluate_results failed.")
        sys.exit(1)

    # 4. Phase 2: Handle Closure/Factory Pattern vs Simple Function
    # The provided evaluate_results function does not return a callable; it performs analysis.
    # Therefore, we treat this as Scenario A (Simple Function).
    # However, since the function mostly prints and saves a plot, 'output' might be None.
    # We verify that the return value matches the expected return value (likely None).
    
    if inner_paths:
        print("Detected inner data files, but evaluate_results is not a factory function in the provided context.")
        print("Proceeding with simple return value comparison.")

    # 5. Verification
    print("Verifying results...")
    passed, msg = recursive_check(expected_outer_output, actual_result)

    if not passed:
        print(f"FAILED: Result mismatch. {msg}")
        sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()