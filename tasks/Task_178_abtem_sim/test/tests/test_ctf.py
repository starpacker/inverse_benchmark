import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_ctf import ctf
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/abtem_sim_sandbox_sandbox/run_code/std_data/standard_data_ctf.pkl']
    
    # Filter paths to identify outer and inner data files
    outer_path = None
    inner_path = None
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_path = path
        elif basename == 'standard_data_ctf.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_ctf.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
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
        agent_result = ctf(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute ctf with outer data")
        print(traceback.format_exc())
        sys.exit(1)
    
    # Phase 2: Determine scenario and verify
    if inner_path is not None and os.path.exists(inner_path):
        # Scenario B: Factory/Closure pattern
        # The result should be callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable from ctf, got {type(agent_result)}")
            sys.exit(1)
        
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
            result = agent_result(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute agent_result (inner call)")
            print(traceback.format_exc())
            sys.exit(1)
    else:
        # Scenario A: Simple function
        result = agent_result
        expected = outer_data.get('output')
    
    # Verification
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception")
        print(traceback.format_exc())
        sys.exit(1)
    
    if not passed:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)
    else:
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()