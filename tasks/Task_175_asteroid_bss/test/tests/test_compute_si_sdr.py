import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_compute_si_sdr import compute_si_sdr
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/asteroid_bss_sandbox_sandbox/run_code/std_data/standard_data_compute_si_sdr.pkl']
    
    # Filter paths to find outer and inner data files
    outer_path = None
    inner_path = None
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_path = path
        elif basename == 'standard_data_compute_si_sdr.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_compute_si_sdr.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    
    # Execute the function
    try:
        result = compute_si_sdr(*outer_args, **outer_kwargs)
        print(f"Function executed successfully")
    except Exception as e:
        print(f"ERROR: Failed to execute compute_si_sdr: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine if this is a factory pattern or simple function
    if inner_path is not None and os.path.exists(inner_path):
        # Scenario B: Factory/Closure Pattern
        print(f"Detected factory pattern, loading inner data from: {inner_path}")
        
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        
        # The result from Phase 1 should be callable
        if not callable(result):
            print(f"ERROR: Expected callable operator from factory, got {type(result)}")
            sys.exit(1)
        
        try:
            actual_result = result(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected = inner_data.get('output')
    else:
        # Scenario A: Simple Function
        print("Detected simple function pattern")
        actual_result = result
        expected = outer_data.get('output')
    
    # Comparison
    try:
        passed, msg = recursive_check(expected, actual_result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if not passed:
        print(f"TEST FAILED: {msg}")
        print(f"Expected: {expected}")
        print(f"Actual: {actual_result}")
        sys.exit(1)
    else:
        print("TEST PASSED")
        sys.exit(0)

if __name__ == '__main__':
    main()