import sys
import os
import dill
import numpy as np
import traceback

from agent_compute_psnr import compute_psnr
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/asteroid_bss_sandbox_sandbox/run_code/std_data/standard_data_compute_psnr.pkl']
    
    # Filter paths to find outer and inner data files
    outer_path = None
    inner_path = None
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"Warning: Path does not exist: {path}")
            continue
        
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_path = path
        elif basename == 'standard_data_compute_psnr.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_compute_psnr.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    # Execute the function
    try:
        result = compute_psnr(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute compute_psnr: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine expected output based on scenario
    if inner_path is not None:
        # Scenario B: Factory/Closure pattern
        # The result should be callable
        if not callable(result):
            print(f"ERROR: Expected callable from compute_psnr, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Load inner data
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected = inner_data.get('output', None)
        
        # Execute the operator
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute agent_operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function
        expected = outer_output
    
    # Phase 2: Verification
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if not passed:
        print(f"TEST FAILED: {msg}")
        print(f"Expected: {expected}")
        print(f"Actual: {result}")
        sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()