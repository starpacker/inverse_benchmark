import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_lorentzian_peak import lorentzian_peak
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/rampy_unmix_sandbox_sandbox/run_code/std_data/standard_data_lorentzian_peak.pkl']
    
    # Filter paths to find outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename or 'parent_' in filename:
            inner_paths.append(path)
        elif filename == 'standard_data_lorentzian_peak.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_lorentzian_peak.pkl)")
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
        agent_result = lorentzian_peak(*outer_args, **outer_kwargs)
        print(f"Successfully called lorentzian_peak with outer args")
    except Exception as e:
        print(f"ERROR: Failed to execute lorentzian_peak: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if this is a factory/closure pattern or simple function
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        # The agent_result should be callable
        if not callable(agent_result):
            print("ERROR: Expected callable operator from lorentzian_peak but got non-callable")
            sys.exit(1)
        
        # Load inner data and execute the operator
        inner_path = inner_paths[0]  # Use first inner path
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            print(f"Loaded inner data from: {inner_path}")
        except Exception as e:
            print(f"ERROR: Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected = inner_data.get('output')
        
        try:
            result = agent_result(*inner_args, **inner_kwargs)
            print("Successfully executed the operator with inner args")
        except Exception as e:
            print(f"ERROR: Failed to execute operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple Function
        result = agent_result
        expected = outer_output
    
    # Comparison
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR: recursive_check failed: {e}")
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