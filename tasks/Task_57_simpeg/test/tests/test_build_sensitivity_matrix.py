import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_build_sensitivity_matrix import build_sensitivity_matrix

# Import verification utility
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/simpeg_sandbox_sandbox/run_code/std_data/standard_data_build_sensitivity_matrix.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_build_sensitivity_matrix.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_build_sensitivity_matrix.pkl)")
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
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    
    # Run the function to get the operator/result
    try:
        agent_operator = build_sensitivity_matrix(*outer_args, **outer_kwargs)
        print("Successfully called build_sensitivity_matrix with outer data")
    except Exception as e:
        print(f"ERROR: Failed to execute build_sensitivity_matrix: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Execution & Verification
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        # Load inner data and execute the returned operator
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
        
        # Verify operator is callable
        if not callable(agent_operator):
            print("ERROR: agent_operator is not callable but inner data exists")
            sys.exit(1)
        
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
            print("Successfully executed agent_operator with inner data")
        except Exception as e:
            print(f"ERROR: Failed to execute agent_operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function
        # The result from Phase 1 IS the result
        result = agent_operator
        expected = outer_data.get('output')
    
    # Comparison
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
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