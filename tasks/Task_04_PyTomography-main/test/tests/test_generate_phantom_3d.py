import sys
import os
import dill
import torch
import numpy as np
import traceback
from agent_generate_phantom_3d import generate_phantom_3d
from verification_utils import recursive_check

def test_generate_phantom_3d():
    # 1. Configuration
    data_paths = ['/data/yjh/PyTomography-main_sandbox/run_code/std_data/standard_data_generate_phantom_3d.pkl']
    
    # 2. Identify Data Files
    # Based on the decorator logic, Scenario A applies here: the function returns a tensor directly,
    # not a callable closure. We expect only the primary data file.
    primary_path = None
    for p in data_paths:
        if 'standard_data_generate_phantom_3d.pkl' in p:
            primary_path = p
            break

    if not primary_path:
        print("Error: Primary data file 'standard_data_generate_phantom_3d.pkl' not found.")
        sys.exit(1)

    # 3. Load Data
    try:
        with open(primary_path, 'rb') as f:
            data = dill.load(f)
        
        args = data['args']
        kwargs = data['kwargs']
        expected_output = data['output']
    except Exception as e:
        print(f"Error loading data file {primary_path}: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Execute Function
    print(f"Running generate_phantom_3d with args: {len(args)} items, kwargs: {list(kwargs.keys())}")
    try:
        # Re-set seed if deterministic behavior is required by the function (though logic seems deterministic given inputs)
        torch.manual_seed(42)
        actual_output = generate_phantom_3d(*args, **kwargs)
    except Exception as e:
        print(f"Error executing generate_phantom_3d: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 5. Verification
    try:
        passed, msg = recursive_check(expected_output, actual_output)
    except Exception as e:
        print(f"Error during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    test_generate_phantom_3d()