import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_generate_rotations import generate_rotations
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/cryodrgn_sandbox_sandbox/run_code/std_data/standard_data_generate_rotations.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        else:
            outer_path = p
    
    if outer_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)
    
    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 1: Run the function
    try:
        agent_result = generate_rotations(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute generate_rotations: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        # agent_result should be callable
        if not callable(agent_result):
            print("ERROR: Expected callable from generate_rotations but got non-callable.")
            sys.exit(1)
        
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute inner call: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED (inner): {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        expected = outer_output
        result = agent_result
        
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    sys.exit(0)

if __name__ == '__main__':
    main()