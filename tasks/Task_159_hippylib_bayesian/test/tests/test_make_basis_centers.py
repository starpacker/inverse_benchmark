import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_make_basis_centers import make_basis_centers
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/hippylib_bayesian_sandbox_sandbox/run_code/std_data/standard_data_make_basis_centers.pkl']
    
    # Filter paths to find outer (main function data) and inner (closure/operator data)
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_make_basis_centers.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find main data file (standard_data_make_basis_centers.pkl)")
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
    
    try:
        agent_operator = make_basis_centers(*outer_args, **outer_kwargs)
        print(f"Successfully called make_basis_centers with args={outer_args}, kwargs={outer_kwargs}")
    except Exception as e:
        print(f"ERROR: Failed to execute make_basis_centers: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Execution & Verification
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        # The agent_operator is callable and we need to execute it with inner data
        if not callable(agent_operator):
            print("ERROR: Expected agent_operator to be callable for closure pattern")
            sys.exit(1)
        
        for inner_path in inner_paths:
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
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Successfully executed agent_operator with inner args")
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Comparison
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                print(f"Inner test passed for {inner_path}")
            except Exception as e:
                print(f"ERROR: Comparison failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function - the result from Phase 1 IS the result
        result = agent_operator
        expected = outer_data.get('output')
        
        # Comparison
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Comparison failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()