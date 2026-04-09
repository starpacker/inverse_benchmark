import sys
import os
import dill
import numpy as np
import traceback

from agent_compute_snr_at_position import compute_snr_at_position
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/pynpoint_hci_sandbox_sandbox/run_code/std_data/standard_data_compute_snr_at_position.pkl']
    
    # Filter paths to identify outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"Warning: Path does not exist: {path}")
            continue
        
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_compute_snr_at_position.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_compute_snr_at_position.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Successfully loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    # Phase 2: Execute the function
    try:
        result = compute_snr_at_position(*outer_args, **outer_kwargs)
        print("Successfully executed compute_snr_at_position")
    except Exception as e:
        print(f"ERROR: Failed to execute compute_snr_at_position: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if result is callable (factory pattern - Scenario B)
    if callable(result) and len(inner_paths) > 0:
        print("Detected factory/closure pattern")
        agent_operator = result
        
        # Load inner data and execute
        inner_path = inner_paths[0]
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            print(f"Successfully loaded inner data from: {inner_path}")
        except Exception as e:
            print(f"ERROR: Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected_output = inner_data.get('output')
        
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
            print("Successfully executed inner operator")
        except Exception as e:
            print(f"ERROR: Failed to execute inner operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Phase 3: Verification
    try:
        passed, msg = recursive_check(expected_output, result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if not passed:
        print(f"TEST FAILED: {msg}")
        print(f"Expected: {expected_output}")
        print(f"Got: {result}")
        sys.exit(1)
    else:
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()