import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_uv_to_grid_indices import uv_to_grid_indices
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/priism_radio_sandbox_sandbox/run_code/std_data/standard_data_uv_to_grid_indices.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_uv_to_grid_indices.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_uv_to_grid_indices.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator/result
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
    
    # Execute the function with outer args
    try:
        agent_result = uv_to_grid_indices(*outer_args, **outer_kwargs)
        print(f"Successfully executed uv_to_grid_indices with outer args")
    except Exception as e:
        print(f"ERROR: Failed to execute uv_to_grid_indices: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if this is a factory pattern (inner paths exist)
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        # The agent_result should be callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable from factory pattern, got {type(agent_result)}")
            sys.exit(1)
        
        agent_operator = agent_result
        
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Successfully executed inner operator")
            except Exception as e:
                print(f"ERROR: Failed to execute inner operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verification
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED for inner data {inner_path}")
                print(f"Failure message: {msg}")
                sys.exit(1)
            print(f"Inner test passed for: {inner_path}")
        
        print("TEST PASSED")
        sys.exit(0)
    else:
        # Scenario A: Simple Function
        # Compare agent_result with outer_output
        result = agent_result
        expected = outer_output
        
        passed, msg = recursive_check(expected, result)
        if not passed:
            print(f"TEST FAILED")
            print(f"Failure message: {msg}")
            sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()