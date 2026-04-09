import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/asteroid_bss_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Separate outer and inner paths
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
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    
    # Phase 2: Execute function
    try:
        print(f"[INFO] Executing load_and_preprocess_data with args={outer_args}, kwargs={outer_kwargs}")
        agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("[INFO] Function execution completed successfully")
    except Exception as e:
        print(f"ERROR: Failed to execute load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if this is Scenario A or B
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        # The agent_result should be callable
        if not callable(agent_result):
            print("ERROR: Expected callable operator from load_and_preprocess_data, but got non-callable")
            sys.exit(1)
        
        # Load inner data and execute
        inner_path = inner_paths[0]  # Use first inner path
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            print(f"[INFO] Loaded inner data from: {inner_path}")
        except Exception as e:
            print(f"ERROR: Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        
        try:
            result = agent_result(*inner_args, **inner_kwargs)
            print("[INFO] Inner function execution completed successfully")
        except Exception as e:
            print(f"ERROR: Failed to execute inner operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected = inner_data.get('output')
    else:
        # Scenario A: Simple function
        result = agent_result
        expected = outer_data.get('output')
    
    # Phase 3: Comparison
    try:
        passed, msg = recursive_check(expected, result)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()