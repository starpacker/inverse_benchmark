import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_build_step import build_step
from verification_utils import recursive_check

# Data paths provided
data_paths = ['/data/yjh/empymod_sandbox_sandbox/run_code/std_data/standard_data_build_step.pkl']

def main():
    """Main test function."""
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_build_step.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_build_step.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run function
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
        result = build_step(*outer_args, **outer_kwargs)
        print("Successfully executed build_step")
    except Exception as e:
        print(f"ERROR: Failed to execute build_step: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if we have inner paths (Scenario B: Factory/Closure Pattern)
    if inner_paths:
        # Scenario B: The result should be callable (an operator)
        if not callable(result):
            print("ERROR: Expected callable result for factory pattern, but got non-callable")
            sys.exit(1)
        
        agent_operator = result
        print("Result is callable, proceeding with inner data execution")
        
        # Load and execute with inner data
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
                print("Successfully executed agent_operator with inner data")
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
                print("Inner data verification passed")
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function - compare result directly
        expected = outer_data.get('output')
        
        try:
            passed, msg = recursive_check(expected, result)
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