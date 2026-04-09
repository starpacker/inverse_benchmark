import sys
import os
import dill
import torch
import numpy as np
import traceback
from agent_dim_match import dim_match
from verification_utils import recursive_check

# Hardcoded paths as provided in the instructions
data_paths = ['/data/yjh/mripy-master_sandbox/run_code/std_data/standard_data_dim_match.pkl']

def main():
    print("Starting test_dim_match.py...")
    
    # 1. Identify the Data File
    # Since dim_match is a simple utility function returning values (not a closure),
    # we expect only the 'outer' standard data file.
    outer_path = None
    for path in data_paths:
        if 'standard_data_dim_match.pkl' in path:
            outer_path = path
            break
            
    if not outer_path:
        print("Error: Could not find 'standard_data_dim_match.pkl' in provided paths.")
        sys.exit(1)
        
    print(f"Loading data from: {outer_path}")
    
    # 2. Load Data
    try:
        with open(outer_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    # Extract inputs and expected output
    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_output = data.get('output', None)
    
    print(f"Loaded args length: {len(args)}")
    print(f"Loaded kwargs keys: {list(kwargs.keys())}")
    
    # 3. Execute Target Function
    try:
        print("Executing dim_match...")
        actual_output = dim_match(*args, **kwargs)
    except Exception as e:
        print(f"Error executing dim_match: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Verify Results
    print("Verifying results...")
    
    # Since dim_match returns simple tuples/shapes, we compare directly against expected output
    is_correct, msg = recursive_check(expected_output, actual_output)
    
    if is_correct:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()