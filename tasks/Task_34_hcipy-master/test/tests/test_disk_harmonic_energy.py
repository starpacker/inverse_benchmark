import sys
import os
import dill
import traceback
import numpy as np

# Handle optional torch import to prevent ModuleNotFoundError
try:
    import torch
except ImportError:
    torch = None

from agent_disk_harmonic_energy import disk_harmonic_energy
from verification_utils import recursive_check

def test_disk_harmonic_energy():
    """
    Unit test for disk_harmonic_energy function.
    Scenario: Simple Function (returns scalar).
    """
    
    # Paths provided in instructions
    data_paths = ['/data/yjh/hcipy-master_sandbox/run_code/std_data/standard_data_disk_harmonic_energy.pkl']
    
    # Filter for the main data file
    file_path = None
    for p in data_paths:
        if 'standard_data_disk_harmonic_energy.pkl' in p:
            file_path = p
            break
            
    if not file_path or not os.path.exists(file_path):
        print(f"Test Skipped: Data file not found at {file_path}")
        sys.exit(0)

    try:
        # Load data
        with open(file_path, 'rb') as f:
            data = dill.load(f)
        
        args = data.get('args', [])
        kwargs = data.get('kwargs', {})
        expected_result = data.get('output')
        
        print(f"Running test with args: {args}, kwargs: {kwargs}")
        
        # Execute target function
        actual_result = disk_harmonic_energy(*args, **kwargs)
        
        # Verify results
        passed, msg = recursive_check(expected_result, actual_result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            print(f"Expected: {expected_result}")
            print(f"Actual:   {actual_result}")
            sys.exit(1)

    except Exception as e:
        print(f"TEST FAILED with Exception: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_disk_harmonic_energy()