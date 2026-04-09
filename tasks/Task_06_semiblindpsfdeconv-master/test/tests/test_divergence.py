import sys
import os
import dill
import numpy as np
import torch
import traceback
from agent_divergence import divergence
from verification_utils import recursive_check

# List of data paths provided for the test
data_paths = ['/data/yjh/semiblindpsfdeconv-master_sandbox/run_code/std_data/standard_data_divergence.pkl']

def load_data(path):
    """Helper to load dill pickle files securely."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return dill.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def test_divergence():
    print("----------------------------------------------------------------")
    print("Starting Unit Test for: divergence")
    print("----------------------------------------------------------------")

    # 1. Identify the primary data file
    # In this scenario, 'divergence' appears to be a direct calculation function (Scenario A),
    # not a factory. We look for the standard data file.
    primary_data_path = None
    for path in data_paths:
        if 'standard_data_divergence.pkl' in path:
            primary_data_path = path
            break

    if not primary_data_path:
        print("Skipping test: standard_data_divergence.pkl not found in provided paths.")
        # If no data is found, we can't test, but it's not strictly a code failure.
        sys.exit(0)

    # 2. Load Data
    print(f"Loading data from: {primary_data_path}")
    data = load_data(primary_data_path)
    if data is None:
        print("Failed to load data.")
        sys.exit(1)

    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_output = data.get('output')

    # 3. Execution
    print("Executing 'divergence' with loaded inputs...")
    try:
        # Direct execution since divergence(F) returns the result directly
        actual_result = divergence(*args, **kwargs)
    except Exception as e:
        print("Execution failed with error:")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verification
    print("Verifying results...")
    is_correct, fail_msg = recursive_check(expected_output, actual_result)

    if is_correct:
        print("TEST PASSED: Output matches expected results.")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {fail_msg}")
        # Debug info
        print(f"Expected Type: {type(expected_output)}")
        print(f"Actual Type:   {type(actual_result)}")
        sys.exit(1)

if __name__ == "__main__":
    test_divergence()