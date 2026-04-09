import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent__lucas_kanade_refine import _lucas_kanade_refine
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/pyidi_dic_sandbox_sandbox/run_code/std_data/standard_data__lucas_kanade_refine.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data__lucas_kanade_refine.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data__lucas_kanade_refine.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}")
        print(traceback.format_exc())
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    # Execute the function
    try:
        result = _lucas_kanade_refine(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute _lucas_kanade_refine with outer data")
        print(traceback.format_exc())
        sys.exit(1)
    
    # Check if there are inner paths (Scenario B: Factory/Closure pattern)
    if inner_paths:
        # Scenario B: The result is an operator/closure
        agent_operator = result
        
        if not callable(agent_operator):
            print("ERROR: Expected a callable operator from _lucas_kanade_refine but got non-callable")
            sys.exit(1)
        
        # Load inner data and execute
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
            expected_output = inner_data.get('output')
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator with inner data from {inner_path}")
                print(traceback.format_exc())
                sys.exit(1)
            
            # Verify result
            try:
                passed, msg = recursive_check(expected_output, result)
            except Exception as e:
                print(f"ERROR: Verification failed with exception")
                print(traceback.format_exc())
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED for inner data {inner_path}")
                print(f"Verification message: {msg}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    else:
        # Scenario A: Simple function, result is the output
        try:
            passed, msg = recursive_check(expected_output, result)
        except Exception as e:
            print(f"ERROR: Verification failed with exception")
            print(traceback.format_exc())
            sys.exit(1)
        
        if not passed:
            print("TEST FAILED")
            print(f"Verification message: {msg}")
            sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()