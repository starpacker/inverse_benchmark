import sys
import os
import dill
import numpy as np
import traceback

# Add path for imports
sys.path.insert(0, '/home/yjh/oopao_sh_sandbox/run_code')

from agent_integral_controller_explicit import integral_controller_explicit
from verification_utils import recursive_check

def main():
    """Main test function for integral_controller_explicit"""
    
    # Data paths provided
    data_paths = ['/home/yjh/oopao_sh_sandbox/run_code/std_data/standard_data_integral_controller_explicit.pkl']
    
    # Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_integral_controller_explicit.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_integral_controller_explicit.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}")
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    # Scenario A: Simple function call (no inner data)
    if not inner_paths:
        try:
            # Run the function directly
            result = integral_controller_explicit(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute integral_controller_explicit")
            print(f"Exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Compare result with expected output
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
    
    # Scenario B: Factory/Closure pattern (with inner data)
    else:
        try:
            # Create the operator/closure
            agent_operator = integral_controller_explicit(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to create operator from integral_controller_explicit")
            print(f"Exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify operator is callable
        if not callable(agent_operator):
            print(f"ERROR: Result is not callable. Got type: {type(agent_operator)}")
            sys.exit(1)
        
        # Process each inner data file
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute operator with inner data")
                print(f"Exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                passed, msg = recursive_check(inner_expected, result)
                if not passed:
                    print(f"TEST FAILED for {inner_path}: {msg}")
                    sys.exit(1)
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()