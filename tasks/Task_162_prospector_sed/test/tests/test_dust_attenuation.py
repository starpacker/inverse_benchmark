import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_dust_attenuation import dust_attenuation
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/prospector_sed_sandbox_sandbox/run_code/std_data/standard_data_dust_attenuation.pkl']

def main():
    # Filter paths to find outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_dust_attenuation.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_dust_attenuation.pkl)")
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
    expected_output = outer_data.get('output')
    
    # Execute the function
    try:
        result = dust_attenuation(*outer_args, **outer_kwargs)
        print("Successfully executed dust_attenuation")
    except Exception as e:
        print(f"ERROR: Failed to execute dust_attenuation: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Check if this is a factory pattern (result is callable) and inner data exists
    if len(inner_paths) > 0 and callable(result):
        # Scenario B: Factory/Closure Pattern
        print("Detected factory pattern with inner data files")
        agent_operator = result
        
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
            expected_output = inner_data.get('output')
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed the agent operator with inner data")
            except Exception as e:
                print(f"ERROR: Failed to execute agent operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify result
            try:
                passed, msg = recursive_check(expected_output, result)
                if not passed:
                    print(f"TEST FAILED for inner data {inner_path}: {msg}")
                    sys.exit(1)
                print(f"Verification passed for inner data: {inner_path}")
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function - compare result directly with outer expected output
        try:
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()