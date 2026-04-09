import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/crackpy_sif_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']

def main():
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct the operator/result
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
        agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute load_and_preprocess_data with outer args: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine if this is Scenario A or B
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        # The agent_result should be callable
        if not callable(agent_result):
            print("ERROR: Expected agent_result to be callable for Scenario B, but it is not.")
            sys.exit(1)
        
        # Process each inner path
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
                result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute operator with inner args: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"ERROR: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED for inner path {inner_path}: {msg}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    else:
        # Scenario A: Simple function - result is the output
        expected = outer_data.get('output')
        result = agent_result
        
        # Compare results
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()