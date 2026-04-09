import sys
import os
import dill
import numpy as np
import traceback

from agent_compute_concentration_re import compute_concentration_re
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/spectrochempy_mcr_sandbox_sandbox/run_code/std_data/standard_data_compute_concentration_re.pkl']

def main():
    # Determine outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_compute_concentration_re.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_compute_concentration_re.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and call function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    try:
        result = compute_concentration_re(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute compute_concentration_re: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if we have inner paths (Scenario B: Factory/Closure Pattern)
    if inner_paths:
        # Scenario B: The result is a callable operator
        if not callable(result):
            print(f"ERROR: Expected callable operator but got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Load inner data and execute
        inner_path = inner_paths[0]  # Use first inner path
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load inner data file: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected_output = inner_data.get('output')
        
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute agent_operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Phase 2: Verification
    try:
        passed, msg = recursive_check(expected_output, result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
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