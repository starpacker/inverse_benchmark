import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_to_db import to_db
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/acoular_beamforming_sandbox_sandbox/run_code/std_data/standard_data_to_db.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_to_db.pkl':
            outer_path = path
    
    # Phase 1: Load outer data and reconstruct operator
    if outer_path is None:
        print("ERROR: No outer data file found (standard_data_to_db.pkl)")
        sys.exit(1)
    
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}")
        print(traceback.format_exc())
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    
    try:
        # Call the function with outer args/kwargs
        agent_result = to_db(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute to_db with outer args")
        print(traceback.format_exc())
        sys.exit(1)
    
    # Phase 2: Determine scenario and verify
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        # The result from Phase 1 should be callable
        if not callable(agent_result):
            print("ERROR: Expected callable operator from to_db, but got non-callable")
            sys.exit(1)
        
        agent_operator = agent_result
        
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
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator with inner args")
                print(traceback.format_exc())
                sys.exit(1)
            
            # Compare results
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED for {inner_path}")
                print(msg)
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    else:
        # Scenario A: Simple function
        # The result from Phase 1 IS the final result
        result = agent_result
        expected = outer_data.get('output')
        
        # Compare results
        passed, msg = recursive_check(expected, result)
        if not passed:
            print("TEST FAILED")
            print(msg)
            sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()