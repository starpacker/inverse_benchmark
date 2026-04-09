import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_compute_metrics_linear import compute_metrics_linear
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/acoular_beamforming_sandbox_sandbox/run_code/std_data/standard_data_compute_metrics_linear.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_compute_metrics_linear.pkl':
            outer_path = path
    
    # If no specific outer path found, use the first one
    if outer_path is None and len(data_paths) > 0:
        outer_path = data_paths[0]
    
    if outer_path is None:
        print("ERROR: No data file found")
        sys.exit(1)
    
    # Load outer data
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
    outer_output = outer_data.get('output', None)
    
    # Phase 1: Execute the function with outer args
    try:
        result = compute_metrics_linear(*outer_args, **outer_kwargs)
        print(f"Function executed successfully")
    except Exception as e:
        print(f"ERROR: Failed to execute compute_metrics_linear: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if we have inner paths (Scenario B: Factory/Closure Pattern)
    if inner_paths:
        # Scenario B: The result should be a callable operator
        if not callable(result):
            print(f"ERROR: Expected callable operator but got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Load inner data
        inner_path = inner_paths[0]
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
        expected = inner_data.get('output', None)
        
        # Execute the operator with inner args
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
            print(f"Operator executed successfully")
        except Exception as e:
            print(f"ERROR: Failed to execute operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function - result is the output
        expected = outer_output
    
    # Comparison
    try:
        passed, msg = recursive_check(expected, result)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            print(f"Expected: {expected}")
            print(f"Got: {result}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Comparison failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()