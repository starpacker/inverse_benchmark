import sys
import os
import dill
import numpy as np
import traceback

# Add the current directory to sys.path to ensure local imports work
sys.path.append(os.getcwd())

# Import the target function
from agent_forward_operator import forward_operator
# Import verification utility
from verification_utils import recursive_check

def test_forward_operator():
    """
    Unit test for forward_operator using captured standard data.
    """
    print("----------------------------------------------------------------")
    print("Running Unit Test: test_forward_operator")
    print("----------------------------------------------------------------")

    # 1. Define Data Paths
    # Based on the provided analysis, we only have one file.
    # This implies Scenario A: standard function execution.
    data_paths = ['/data/yjh/MPIRF-master_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    outer_path = None
    inner_path = None

    # Categorize paths (though in this specific case, we expect only outer)
    for path in data_paths:
        if "parent_function" in path:
            inner_path = path
        else:
            outer_path = path

    if not outer_path or not os.path.exists(outer_path):
        print(f"FAILED: Main data file not found or path list is empty: {outer_path}")
        sys.exit(1)

    # 2. Load Data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded data from {outer_path}")
    except Exception as e:
        print(f"FAILED: Could not load data file. Error: {e}")
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_result = outer_data.get('output', None)

    # 3. Execution
    try:
        print("Executing forward_operator with loaded arguments...")
        # Since no inner path was found in the provided list, we treat this as a direct function call
        actual_result = forward_operator(*outer_args, **outer_kwargs)
        
        # Check if the result is a callable (Closure Pattern check)
        # If the function actually returns another function but we only have the outer data file provided in the prompt,
        # we can't test the inner part. However, looking at the source code of forward_operator provided,
        # it returns (system_matrix_freq, voltage_freq), which are numpy arrays, not a callable.
        # So this is definitely Scenario A.

    except Exception as e:
        print("FAILED: Execution raised an exception.")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verification
    try:
        passed, msg = recursive_check(expected_result, actual_result)
        if passed:
            print("TEST PASSED: Output matches expected standard data.")
            sys.exit(0)
        else:
            print(f"FAILED: Output mismatch.\nDetails: {msg}")
            sys.exit(1)
    except Exception as e:
        print(f"FAILED: Verification process raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_forward_operator()