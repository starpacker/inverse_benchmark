import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_forward_operator import forward_operator
from verification_utils import recursive_check

def main():
    """Main test function for forward_operator."""
    
    # Data paths provided
    data_paths = ['/data/yjh/suite2p_spike_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Separate outer and inner data paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_forward_operator.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run forward_operator
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
    outer_expected = outer_data.get('output')
    
    print(f"Outer function name: {outer_data.get('func_name', 'N/A')}")
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute forward_operator with outer data
    try:
        result = forward_operator(*outer_args, **outer_kwargs)
        print("Successfully executed forward_operator")
    except Exception as e:
        print(f"ERROR: Failed to execute forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if result is callable (factory pattern - Scenario B)
    if inner_paths and callable(result):
        # Scenario B: Factory/Closure Pattern
        print("Detected factory/closure pattern (Scenario B)")
        agent_operator = result
        
        # Load inner data
        inner_path = inner_paths[0]  # Use the first inner path
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
        
        print(f"Inner function name: {inner_data.get('func_name', 'N/A')}")
        print(f"Inner args count: {len(inner_args)}")
        print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
        
        # Execute the operator with inner data
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
            print("Successfully executed inner operator")
        except Exception as e:
            print(f"ERROR: Failed to execute inner operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function
        print("Detected simple function pattern (Scenario A)")
        expected = outer_expected
    
    # Phase 2: Verification
    print("\n--- Verification Phase ---")
    print(f"Expected type: {type(expected)}")
    print(f"Result type: {type(result)}")
    
    if isinstance(expected, np.ndarray):
        print(f"Expected shape: {expected.shape}, dtype: {expected.dtype}")
    if isinstance(result, np.ndarray):
        print(f"Result shape: {result.shape}, dtype: {result.dtype}")
    
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if passed:
        print("\n========================================")
        print("TEST PASSED")
        print("========================================")
        sys.exit(0)
    else:
        print("\n========================================")
        print("TEST FAILED")
        print(f"Mismatch details: {msg}")
        print("========================================")
        sys.exit(1)

if __name__ == "__main__":
    main()