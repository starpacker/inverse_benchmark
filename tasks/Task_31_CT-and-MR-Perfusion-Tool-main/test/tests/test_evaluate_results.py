import sys
import os
import dill
import traceback
import numpy as np
import warnings

# Safe import for torch to handle environments where it might be missing
# but potentially referenced in pickle metadata
try:
    import torch
except ImportError:
    torch = None

# Add the directory containing the agent code to sys.path so we can import the module
sys.path.append('/data/yjh/CT-and-MR-Perfusion-Tool-main_sandbox/run_code')

# Attempt to import the target function
try:
    from agent_evaluate_results import evaluate_results
except ImportError as e:
    print(f"Failed to import evaluate_results: {e}")
    sys.exit(1)

# Attempt to import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    # Fallback implementation if verification_utils is missing
    def recursive_check(expected, actual):
        if expected is None and actual is None:
            return True, "Both are None"
        if isinstance(expected, (int, float, str, bool)):
            if expected == actual:
                return True, "Values match"
            return False, f"Expected {expected}, got {actual}"
        # For this specific test (evaluate_results returns None), complex checking might not be hit often
        # but basic type check is good practice.
        if type(expected) != type(actual):
             return False, f"Type mismatch: expected {type(expected)}, got {type(actual)}"
        return True, "Structure matches (simplified check)"

def main():
    # 1. Configuration
    data_paths = ['/data/yjh/CT-and-MR-Perfusion-Tool-main_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # 2. Identify the Data File
    # In this context, we are looking for the primary input file for 'evaluate_results'
    target_file = None
    for path in data_paths:
        if 'standard_data_evaluate_results.pkl' in path:
            target_file = path
            break
            
    if not target_file:
        print("Error: Could not find 'standard_data_evaluate_results.pkl' in provided paths.")
        sys.exit(1)
        
    print(f"Loading data from: {target_file}")

    # 3. Load Data
    try:
        with open(target_file, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Extract Arguments and Expected Output
    # The pickle is expected to contain: {'func_name': ..., 'args': ..., 'kwargs': ..., 'output': ...}
    try:
        args = data.get('args', [])
        kwargs = data.get('kwargs', {})
        expected_output = data.get('output', None)
        
        # Scenario A: Simple Function Execution
        # We run the function directly with the loaded args.
        print(f"Executing evaluate_results with {len(args)} args and {len(kwargs)} kwargs...")
        
        # Execute the function
        actual_result = evaluate_results(*args, **kwargs)
        
    except Exception as e:
        print(f"Execution failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 5. Verification
    print("Verifying results...")
    
    # The 'evaluate_results' function typically returns None (it prints/saves files).
    # We verify that the return value matches the recorded return value (likely None).
    is_correct, msg = recursive_check(expected_output, actual_result)
    
    if is_correct:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        # Detailed debugging info
        print(f"Expected: {expected_output}")
        print(f"Actual:   {actual_result}")
        sys.exit(1)

if __name__ == "__main__":
    main()