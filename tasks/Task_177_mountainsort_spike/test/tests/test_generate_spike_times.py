import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_generate_spike_times import generate_spike_times
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/data/yjh/mountainsort_spike_sandbox_sandbox/run_code/std_data/standard_data_generate_spike_times.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_generate_spike_times.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_generate_spike_times.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output')
    
    try:
        # Execute the function with outer args
        result = generate_spike_times(*outer_args, **outer_kwargs)
        print("Successfully executed generate_spike_times")
    except Exception as e:
        print(f"ERROR executing generate_spike_times: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if result is callable (factory pattern)
    if callable(result) and len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        agent_operator = result
        print("Detected factory pattern - result is callable")
        
        # Load inner data
        inner_path = inner_paths[0]  # Use first inner path
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            print(f"Loaded inner data from: {inner_path}")
        except Exception as e:
            print(f"ERROR loading inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected = inner_data.get('output')
        
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
            print("Successfully executed inner operator")
        except Exception as e:
            print(f"ERROR executing inner operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function
        expected = outer_output
        print("Detected simple function pattern")
    
    # Phase 2: Verification
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR during verification: {e}")
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