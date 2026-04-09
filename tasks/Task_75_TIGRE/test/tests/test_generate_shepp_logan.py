import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_generate_shepp_logan import generate_shepp_logan
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/TIGRE_sandbox_sandbox/run_code/std_data/standard_data_generate_shepp_logan.pkl']
    
    # Separate outer and inner data paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_generate_shepp_logan.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_generate_shepp_logan.pkl)")
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
        agent_result = generate_shepp_logan(*outer_args, **outer_kwargs)
        print("Successfully executed generate_shepp_logan with outer args")
    except Exception as e:
        print(f"ERROR: Failed to execute generate_shepp_logan: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if this is Scenario A or B
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        # The result should be callable
        if not callable(agent_result):
            print("ERROR: Expected callable result for factory pattern but got non-callable")
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
                result = agent_result(*inner_args, **inner_kwargs)
                print("Successfully executed inner function")
            except Exception as e:
                print(f"ERROR: Failed to execute inner function: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify result
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                print(f"Inner test passed for: {inner_path}")
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple Function
        expected = outer_data.get('output')
        result = agent_result
        
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()