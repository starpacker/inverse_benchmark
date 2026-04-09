import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent__visualize_results import _visualize_results
from verification_utils import recursive_check

def main():
    """Main test function for _visualize_results"""
    
    # Data paths provided
    data_paths = ['/data/yjh/pfsspy_sandbox_sandbox/run_code/std_data/standard_data__visualize_results.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data__visualize_results.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data__visualize_results.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute the function
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
    expected_output = outer_data.get('output')
    
    # Scenario B: Check if we have inner paths (factory/closure pattern)
    if inner_paths:
        # Execute outer function to get the operator/closure
        try:
            agent_operator = _visualize_results(*outer_args, **outer_kwargs)
            print(f"[INFO] Created agent operator from outer function")
        except Exception as e:
            print(f"ERROR: Failed to execute outer function: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify it's callable
        if not callable(agent_operator):
            print(f"ERROR: Returned operator is not callable, got type: {type(agent_operator)}")
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
        expected_output = inner_data.get('output')
        
        # Execute the operator with inner args
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
            print(f"[INFO] Executed agent operator with inner data")
        except Exception as e:
            print(f"ERROR: Failed to execute agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    else:
        # Scenario A: Simple function - just execute and compare
        try:
            result = _visualize_results(*outer_args, **outer_kwargs)
            print(f"[INFO] Executed _visualize_results directly")
        except Exception as e:
            print(f"ERROR: Failed to execute function: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Phase 2: Verification
    try:
        passed, msg = recursive_check(expected_output, result)
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