import sys
import os
import dill
import numpy as np
import traceback

from agent_clean_sc import clean_sc
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/acoular_beamforming_sandbox_sandbox/run_code/std_data/standard_data_clean_sc.pkl']
    
    # Filter paths to find outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_clean_sc.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_clean_sc.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}")
        print(traceback.format_exc())
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output')
    
    try:
        result = clean_sc(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute clean_sc with outer data")
        print(traceback.format_exc())
        sys.exit(1)
    
    # Check if we have inner data files (Scenario B: Factory/Closure Pattern)
    if inner_paths:
        # Scenario B: The result should be callable, and we need to test it with inner data
        if not callable(result):
            print("ERROR: Expected callable result from clean_sc for factory pattern, but got non-callable")
            sys.exit(1)
        
        agent_operator = result
        
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}")
                print(traceback.format_exc())
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator with inner data from {inner_path}")
                print(traceback.format_exc())
                sys.exit(1)
            
            passed, msg = recursive_check(expected, actual_result)
            if not passed:
                print(f"TEST FAILED for inner data {inner_path}")
                print(msg)
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    else:
        # Scenario A: Simple function - compare result directly with outer_output
        expected = outer_output
        
        passed, msg = recursive_check(expected, result)
        if not passed:
            print("TEST FAILED")
            print(msg)
            sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()