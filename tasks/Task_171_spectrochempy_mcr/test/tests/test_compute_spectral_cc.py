import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_compute_spectral_cc import compute_spectral_cc
from verification_utils import recursive_check

# Define data paths
data_paths = ['/data/yjh/spectrochempy_mcr_sandbox_sandbox/run_code/std_data/standard_data_compute_spectral_cc.pkl']

def main():
    # Filter paths to find outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_compute_spectral_cc.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_compute_spectral_cc.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run the function
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
    outer_output = outer_data.get('output')
    
    try:
        agent_result = compute_spectral_cc(*outer_args, **outer_kwargs)
        print("Successfully executed compute_spectral_cc with outer data")
    except Exception as e:
        print(f"ERROR: Failed to execute compute_spectral_cc: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if this is a factory/closure pattern or simple function
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        # The result should be callable
        if not callable(agent_result):
            print("ERROR: Expected callable result (operator) but got non-callable")
            sys.exit(1)
        
        agent_operator = agent_result
        
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output')
            
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed the operator with inner data")
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            try:
                passed, msg = recursive_check(expected_output, actual_result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                print(f"Verification passed for inner data: {inner_path}")
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple Function
        # Compare the result directly with expected output
        expected_output = outer_output
        actual_result = agent_result
        
        try:
            passed, msg = recursive_check(expected_output, actual_result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            print("Verification passed for simple function call")
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()