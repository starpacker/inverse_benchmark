import sys
import os
import dill
import numpy as np
import traceback

from agent_effective_temperature import effective_temperature
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/prospector_sed_sandbox_sandbox/run_code/std_data/standard_data_effective_temperature.pkl']

def main():
    # Filter paths to find outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"Warning: Path does not exist: {path}")
            continue
        
        basename = os.path.basename(path)
        
        # Check if it's an inner path (contains 'parent_function' or 'parent_')
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        # Check if it's the outer path (exact match for the function)
        elif basename == 'standard_data_effective_temperature.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_effective_temperature.pkl)")
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
    
    # Execute the function
    try:
        result = effective_temperature(*outer_args, **outer_kwargs)
        print("Successfully executed effective_temperature")
    except Exception as e:
        print(f"ERROR: Failed to execute effective_temperature: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine if this is Scenario A or B
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print("Scenario B detected: Factory/Closure pattern")
        
        # Verify result is callable
        if not callable(result):
            print(f"ERROR: Expected callable operator, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Process inner data
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
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed agent_operator")
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            try:
                passed, msg = recursive_check(expected, actual_result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                print(f"Inner test passed for: {inner_path}")
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple Function
        print("Scenario A detected: Simple function")
        
        expected = outer_data.get('output')
        
        # Verify results
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()