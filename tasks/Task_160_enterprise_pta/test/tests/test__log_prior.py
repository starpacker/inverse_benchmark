import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent__log_prior import _log_prior
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/enterprise_pta_sandbox_sandbox/run_code/std_data/standard_data__log_prior.pkl']
    
    # Filter paths to find outer and inner data files
    outer_path = None
    inner_path = None
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_path = path
        elif basename == 'standard_data__log_prior.pkl':
            outer_path = path
    
    # Scenario A: Simple function - only outer path exists
    if outer_path and not inner_path:
        try:
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
        except Exception as e:
            print(f"FAILED: Could not load outer data file: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Extract args and kwargs
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected = outer_data.get('output')
        
        try:
            result = _log_prior(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAILED: Function execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Compare result with expected
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAILED: Verification check failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"FAILED: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)
    
    # Scenario B: Factory/Closure pattern - both outer and inner paths exist
    elif outer_path and inner_path:
        # Phase 1: Reconstruct the operator
        try:
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
        except Exception as e:
            print(f"FAILED: Could not load outer data file: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        try:
            agent_operator = _log_prior(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAILED: Could not create agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not callable(agent_operator):
            print("FAILED: Agent operator is not callable")
            sys.exit(1)
        
        # Phase 2: Load inner data and execute
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"FAILED: Could not load inner data file: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected = inner_data.get('output')
        
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"FAILED: Agent operator execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Compare result with expected
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAILED: Verification check failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"FAILED: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)
    
    else:
        print("FAILED: No valid data paths found")
        sys.exit(1)

if __name__ == "__main__":
    main()