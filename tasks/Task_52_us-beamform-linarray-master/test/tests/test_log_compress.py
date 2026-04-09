import sys
import os
import dill
import numpy as np
import torch
import traceback
from agent_log_compress import log_compress
from verification_utils import recursive_check

def test_log_compress():
    """
    Unit test for log_compress.
    Based on analysis, log_compress returns a result directly (Scenario A), 
    not a callable/closure.
    """
    
    # 1. Define Data Paths
    data_paths = ['/data/yjh/us-beamform-linarray-master_sandbox/run_code/std_data/standard_data_log_compress.pkl']
    
    # Filter for the main data file
    main_data_path = None
    for path in data_paths:
        if path.endswith('standard_data_log_compress.pkl'):
            main_data_path = path
            break
            
    if main_data_path is None:
        print("Error: standard_data_log_compress.pkl not found in provided paths.")
        sys.exit(1)
        
    if not os.path.exists(main_data_path):
        print(f"Error: File does not exist at path: {main_data_path}")
        sys.exit(1)

    # 2. Load Data
    print(f"Loading data from {main_data_path}...")
    try:
        with open(main_data_path, 'rb') as f:
            data_payload = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Extract inputs and expected output
    args = data_payload.get('args', [])
    kwargs = data_payload.get('kwargs', {})
    expected_output = data_payload.get('output')

    # 3. Execute Function
    print("Executing log_compress...")
    try:
        # Since log_compress is a direct processing function (not a factory),
        # we execute it once to get the final result.
        actual_output = log_compress(*args, **kwargs)
    except Exception as e:
        print(f"Execution failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verification
    print("Verifying results...")
    try:
        passed, msg = recursive_check(expected_output, actual_output)
    except Exception as e:
        print(f"Verification process failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    test_log_compress()