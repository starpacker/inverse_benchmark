import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_williams_mode2_ux import williams_mode2_ux
from verification_utils import recursive_check

def main():
    # Define data paths
    data_paths = ['/data/yjh/crackpy_sif_sandbox_sandbox/run_code/std_data/standard_data_williams_mode2_ux.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_path = None
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_path = path
        elif basename == 'standard_data_williams_mode2_ux.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_williams_mode2_ux.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run the function
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
        agent_result = williams_mode2_ux(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute williams_mode2_ux: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine if this is Scenario A or B
    if inner_path is not None and os.path.exists(inner_path):
        # Scenario B: Factory/Closure Pattern
        # Verify agent_result is callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable from williams_mode2_ux, got {type(agent_result)}")
            sys.exit(1)
        
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
            print(f"ERROR: Failed to execute the returned operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple Function
        result = agent_result
        expected = outer_data.get('output')
    
    # Phase 3: Comparison
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if not passed:
        print(f"TEST FAILED: {msg}")
        print(f"Expected type: {type(expected)}")
        print(f"Result type: {type(result)}")
        if isinstance(expected, np.ndarray) and isinstance(result, np.ndarray):
            print(f"Expected shape: {expected.shape}, Result shape: {result.shape}")
            print(f"Expected dtype: {expected.dtype}, Result dtype: {result.dtype}")
        sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()