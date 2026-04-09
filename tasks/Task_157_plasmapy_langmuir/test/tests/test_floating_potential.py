import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_floating_potential import floating_potential
from verification_utils import recursive_check

def main():
    """Main test function for floating_potential."""
    
    # Data paths provided
    data_paths = ['/data/yjh/plasmapy_langmuir_sandbox_sandbox/run_code/std_data/standard_data_floating_potential.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename or 'parent_' in filename:
            inner_paths.append(path)
        elif filename == 'standard_data_floating_potential.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_floating_potential.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and run function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Successfully loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output')
    
    print(f"Outer args: {outer_args}")
    print(f"Outer kwargs: {outer_kwargs}")
    
    # Execute the function
    try:
        result = floating_potential(*outer_args, **outer_kwargs)
        print(f"Successfully executed floating_potential")
    except Exception as e:
        print(f"ERROR: Failed to execute floating_potential: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is a factory pattern (result is callable) or simple function
    if inner_paths and callable(result):
        # Scenario B: Factory/Closure Pattern
        print("Detected factory/closure pattern - proceeding with inner data execution")
        
        agent_operator = result
        
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Successfully loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            print(f"Inner args: {inner_args}")
            print(f"Inner kwargs: {inner_kwargs}")
            
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Successfully executed inner operator")
            except Exception as e:
                print(f"ERROR: Failed to execute inner operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify results
            try:
                passed, msg = recursive_check(expected, actual_result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    print(f"Expected: {expected}")
                    print(f"Actual: {actual_result}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for: {inner_path}")
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple Function
        print("Detected simple function pattern - comparing direct output")
        
        expected = outer_output
        actual_result = result
        
        print(f"Expected output: {expected}")
        print(f"Actual result: {actual_result}")
        
        try:
            passed, msg = recursive_check(expected, actual_result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                print(f"Expected: {expected}")
                print(f"Actual: {actual_result}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()