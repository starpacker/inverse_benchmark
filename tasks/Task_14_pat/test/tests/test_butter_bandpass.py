import sys
import os
import dill
import numpy as np
import traceback

# Add the current directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_butter_bandpass import butter_bandpass
from verification_utils import recursive_check

def main():
    """Main test function for butter_bandpass."""
    
    data_paths = ['/home/yjh/pat_sandbox/run_code/std_data/standard_data_butter_bandpass.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_butter_bandpass.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find standard_data_butter_bandpass.pkl")
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
    
    try:
        agent_operator = butter_bandpass(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute butter_bandpass with outer args/kwargs")
        print(traceback.format_exc())
        sys.exit(1)
    
    # Phase 2: Execution & Verification
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
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
            
            # Verify agent_operator is callable
            if not callable(agent_operator):
                print(f"ERROR: agent_operator is not callable, got type {type(agent_operator)}")
                sys.exit(1)
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator with inner args/kwargs")
                print(traceback.format_exc())
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"ERROR: recursive_check failed")
                print(traceback.format_exc())
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED for {inner_path}")
                print(msg)
                sys.exit(1)
    else:
        # Scenario A: Simple Function - result from Phase 1 IS the result
        result = agent_operator
        expected = outer_data.get('output')
        
        # Compare results
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: recursive_check failed")
            print(traceback.format_exc())
            sys.exit(1)
        
        if not passed:
            print("TEST FAILED")
            print(msg)
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()