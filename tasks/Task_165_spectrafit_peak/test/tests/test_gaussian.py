import sys
import os
import dill
import numpy as np
import traceback

from agent_gaussian import gaussian
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/spectrafit_peak_sandbox_sandbox/run_code/std_data/standard_data_gaussian.pkl']
    
    # Filter paths to find outer and inner data files
    outer_path = None
    inner_path = None
    
    for path in data_paths:
        if os.path.exists(path):
            basename = os.path.basename(path)
            if 'parent_function' in basename:
                inner_path = path
            elif basename == 'standard_data_gaussian.pkl':
                outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_gaussian.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    
    try:
        agent_operator = gaussian(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute gaussian with outer args: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Execution & Verification
    if inner_path is not None:
        # Scenario B: Factory/Closure Pattern
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        
        if not callable(agent_operator):
            print(f"ERROR: agent_operator is not callable, got type: {type(agent_operator)}")
            sys.exit(1)
        
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute agent_operator with inner args: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected = inner_data.get('output')
    else:
        # Scenario A: Simple Function
        result = agent_operator
        expected = outer_data.get('output')
    
    # Comparison
    try:
        passed, msg = recursive_check(expected, result)
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