import sys
import os
import dill
import torch
import numpy as np
import traceback
import warnings
import nibabel as nib

# Force allowlisted globals for dill to handle numpy/torch types correctly
dill.settings['recurse'] = True

# Add current directory to sys.path to ensure local imports work
sys.path.append(os.getcwd())

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def test_evaluate_results():
    data_paths = ['/data/yjh/AMICO-master_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # 1. Identify File Roles
    outer_path = None
    inner_paths = []

    for p in data_paths:
        if "parent_function_evaluate_results" in p:
            inner_paths.append(p)
        elif "standard_data_evaluate_results.pkl" in p:
            outer_path = p

    if not outer_path:
        print("Error: Standard data file 'standard_data_evaluate_results.pkl' not found.")
        sys.exit(1)

    print(f"Loading data from {outer_path}")
    try:
        with open(outer_path, "rb") as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. Reconstruct / Execute
    # The generation code shows that if the result is callable, it returns a decorator.
    # However, looking at the target function `evaluate_results` provided in the reference code:
    # It returns a boolean (True), not a callable.
    # This means we are in Scenario A (Simple Function), despite the robust check logic.
    
    try:
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        expected_result = outer_data.get('output')

        # Since the function relies on an external file 'GT_NDI.nii.gz' which might not exist in the test env,
        # we need to ensure the arguments passed (maps) are valid numpy arrays as per the signature.
        # The pickle should contain the exact arrays used during recording.

        # Execute the function
        print("Executing evaluate_results...")
        actual_result = evaluate_results(*args, **kwargs)

    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Handle Inner Data (Closure/Factory Pattern) - Unlikely here but kept for robustness
    if callable(actual_result) and inner_paths:
        print("Detected callable output. Attempting to run inner execution (Factory Pattern)...")
        # Just take the first inner path found for verification
        inner_path = inner_paths[0]
        try:
            with open(inner_path, "rb") as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_result = inner_data.get('output') # Update expected result to the inner execution's output
            
            actual_result = actual_result(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"Inner execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

    # 4. Verification
    print("Verifying results...")
    passed, msg = recursive_check(expected_result, actual_result)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    test_evaluate_results()