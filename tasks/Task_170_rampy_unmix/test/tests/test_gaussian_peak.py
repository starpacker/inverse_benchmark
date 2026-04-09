import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_gaussian_peak import gaussian_peak
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/rampy_unmix_sandbox_sandbox/run_code/std_data/standard_data_gaussian_peak.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_gaussian_peak.pkl':
            outer_path = path
    
    # Phase 1: Load outer data and reconstruct operator
    if outer_path is None:
        print("ERROR: No outer data file found (standard_data_gaussian_peak.pkl)")
        sys.exit(1)
    
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    
    try:
        agent_result = gaussian_peak(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute gaussian_peak: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is Scenario A or B
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        # agent_result should be callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable from gaussian_peak, got {type(agent_result)}")
            sys.exit(1)
        
        agent_operator = agent_result
        
        # Load inner data and execute
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
    else:
        # Scenario A: Simple function
        result = agent_result
        expected = outer_data.get('output')
        
        # Verify results
        passed, msg = recursive_check(expected, result)
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()