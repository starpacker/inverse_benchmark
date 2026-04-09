import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_canonical_hrf import canonical_hrf
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/mne_nirs_flim_sandbox_sandbox/run_code/std_data/standard_data_canonical_hrf.pkl']
    
    # Determine outer_path and inner_path
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_canonical_hrf.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_canonical_hrf.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
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
    
    # Run the target function
    try:
        agent_result = canonical_hrf(*outer_args, **outer_kwargs)
        print("Successfully called canonical_hrf with outer data")
    except Exception as e:
        print(f"ERROR: Failed to execute canonical_hrf: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and verify
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        inner_path = inner_paths[0]
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
        
        # Verify agent_result is callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable operator, got {type(agent_result)}")
            sys.exit(1)
        
        # Execute the operator with inner data
        try:
            result = agent_result(*inner_args, **inner_kwargs)
            print("Successfully executed agent operator with inner data")
        except Exception as e:
            print(f"ERROR: Failed to execute agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple Function
        result = agent_result
        expected = outer_data.get('output')
        print("Using Scenario A: Simple function comparison")
    
    # Comparison
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR: Verification check failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()