import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add current directory to path so we can import local modules
sys.path.append(os.getcwd())

from agent_normalize import normalize
from verification_utils import recursive_check

def run_test():
    # 1. Define Data Paths
    data_paths = ['/data/yjh/semiblindpsfdeconv-master_sandbox/run_code/std_data/standard_data_normalize.pkl']
    
    # 2. Identify the Main Data File
    # Based on the provided list, there is only one file, which corresponds to Scenario A (Simple Function).
    main_data_path = next((p for p in data_paths if p.endswith('standard_data_normalize.pkl')), None)

    if not main_data_path:
        print("Error: standard_data_normalize.pkl not found in provided paths.")
        sys.exit(1)

    print(f"Loading data from {main_data_path}...")

    # 3. Load Data
    try:
        with open(main_data_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Error loading data file: {e}")
        sys.exit(1)

    # 4. Extract Inputs and Expected Outputs
    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_result = data.get('output')

    print(f"Function inputs loaded: {len(args)} args, {len(kwargs)} kwargs.")

    # 5. Execute Function
    print("Running normalize function...")
    try:
        actual_result = normalize(*args, **kwargs)
    except Exception as e:
        print("Execution failed with error:")
        traceback.print_exc()
        sys.exit(1)

    # 6. Verification
    print("Verifying results...")
    passed, msg = recursive_check(expected_result, actual_result)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()