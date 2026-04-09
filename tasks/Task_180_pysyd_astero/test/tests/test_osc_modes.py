import sys
import os
import dill
import numpy as np
import traceback

from agent_osc_modes import osc_modes
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/pysyd_astero_sandbox_sandbox/run_code/std_data/standard_data_osc_modes.pkl']

def main():
    # Filter paths to find outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_osc_modes.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_osc_modes.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator/result
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
        agent_result = osc_modes(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute osc_modes with outer data")
        print(traceback.format_exc())
        sys.exit(1)
    
    # Phase 2: Check if this is a factory pattern (inner paths exist) or simple function
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        # agent_result should be callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable from osc_modes, got {type(agent_result)}")
            sys.exit(1)
        
        agent_operator = agent_result
        
        # Process each inner path
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
                print(f"ERROR: Failed to execute agent_operator with inner data from {inner_path}")
                print(traceback.format_exc())
                sys.exit(1)
            
            # Verify result
            try:
                passed, msg = recursive_check(expected, result)
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
        # Scenario A: Simple Function
        expected = outer_data.get('output')
        result = agent_result
        
        # Verify result
        try:
            passed, msg = recursive_check(expected, result)
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