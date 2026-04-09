import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_fft2c import fft2c
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/direct_mri_sandbox_sandbox/run_code/std_data/standard_data_fft2c.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_fft2c.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_fft2c.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    # Determine scenario based on inner paths
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        try:
            agent_operator = fft2c(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to create operator from fft2c")
            print(f"Exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify agent_operator is callable
        if not callable(agent_operator):
            print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)
        
        # Phase 2: Load inner data and execute
        inner_path = inner_paths[0]  # Use first inner path
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load inner data from {inner_path}")
            print(f"Exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected = inner_data.get('output', None)
        
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute agent_operator with inner args")
            print(f"Exception: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple Function
        try:
            result = fft2c(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute fft2c")
            print(f"Exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected = outer_output
    
    # Phase 3: Comparison
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if not passed:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)
    else:
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()