import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_compute_metrics_internal import compute_metrics_internal
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/HySUPP_sandbox_sandbox/run_code/std_data/standard_data_compute_metrics_internal.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_compute_metrics_internal.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: No outer data file found (standard_data_compute_metrics_internal.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}")
        print(traceback.format_exc())
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    try:
        # Execute the function
        result = compute_metrics_internal(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute compute_metrics_internal")
        print(traceback.format_exc())
        sys.exit(1)
    
    # Check if this is a factory pattern (result is callable and inner paths exist)
    if callable(result) and not isinstance(result, type) and len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        agent_operator = result
        
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}")
                print(traceback.format_exc())
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output')
            
            try:
                inner_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator with inner data")
                print(traceback.format_exc())
                sys.exit(1)
            
            # Verify inner result
            passed, msg = recursive_check(inner_expected, inner_result)
            if not passed:
                print(f"TEST FAILED (inner execution): {msg}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    else:
        # Scenario A: Simple function - compare result directly
        passed, msg = recursive_check(expected_output, result)
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()