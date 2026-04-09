import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_harvey_comp import harvey_comp
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/pysyd_astero_sandbox_sandbox/run_code/std_data/standard_data_harvey_comp.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_harvey_comp.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_harvey_comp.pkl)")
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
    
    try:
        agent_result = harvey_comp(*outer_args, **outer_kwargs)
        print(f"Executed harvey_comp with outer args")
    except Exception as e:
        print(f"ERROR: Failed to execute harvey_comp: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is a factory pattern (result is callable) or simple function
    if inner_paths and callable(agent_result) and not isinstance(agent_result, type):
        # Scenario B: Factory/Closure pattern
        print("Detected Factory/Closure pattern")
        agent_operator = agent_result
        
        # Load inner data
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
        
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
            print("Executed inner operator")
        except Exception as e:
            print(f"ERROR: Failed to execute inner operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function
        print("Detected Simple function pattern")
        result = agent_result
        expected = outer_data.get('output')
    
    # Phase 2: Verification
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
            if expected.shape == result.shape:
                diff = np.abs(expected - result)
                print(f"Max diff: {np.max(diff)}, Mean diff: {np.mean(diff)}")
        sys.exit(1)
    else:
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()