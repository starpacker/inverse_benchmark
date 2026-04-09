import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_bg_model import bg_model
from verification_utils import recursive_check

def main():
    data_paths = ['/data/yjh/pysyd_astero_sandbox_sandbox/run_code/std_data/standard_data_bg_model.pkl']
    
    # Filter paths to find outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"Warning: Path does not exist: {path}")
            continue
        
        basename = os.path.basename(path)
        # Check if this is an inner data file (contains 'parent_function' or 'parent_')
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        # Check if this is the outer data file (exact match pattern)
        elif basename == 'standard_data_bg_model.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_bg_model.pkl)")
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
    outer_output = outer_data.get('output')
    
    # Check if there are inner paths (Scenario B: Factory/Closure Pattern)
    if inner_paths:
        # Scenario B: Factory pattern
        try:
            agent_operator = bg_model(*outer_args, **outer_kwargs)
            print("Created agent operator from outer data")
        except Exception as e:
            print(f"ERROR: Failed to create agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not callable(agent_operator):
            print("ERROR: agent_operator is not callable")
            sys.exit(1)
        
        # Phase 2: Load inner data and execute
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
                print("Executed agent operator with inner data")
            except Exception as e:
                print(f"ERROR: Failed to execute agent operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verification
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                print(f"Verification passed for inner path: {inner_path}")
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function call
        try:
            result = bg_model(*outer_args, **outer_kwargs)
            print("Executed bg_model with outer data")
        except Exception as e:
            print(f"ERROR: Failed to execute bg_model: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected = outer_output
        
        # Verification
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()