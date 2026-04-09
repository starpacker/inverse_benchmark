import sys
import os
import dill
import traceback

# Add path for imports if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def main():
    """Main test function for load_and_preprocess_data."""
    
    # Data paths provided
    data_paths = ['/data/yjh/hyperspy_eels_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        # Check if this is an inner path (contains parent_function pattern)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"Outer args: {outer_args}")
        print(f"Outer kwargs: {outer_kwargs}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("Executing load_and_preprocess_data with outer args/kwargs...")
        agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print(f"Function executed successfully. Result type: {type(agent_result)}")
        
    except Exception as e:
        print(f"ERROR: Failed to execute load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and verify
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # Verify agent_result is callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable from outer function, got {type(agent_result)}")
            sys.exit(1)
        
        all_passed = True
        for inner_path in inner_paths:
            try:
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print(f"Inner args: {inner_args}")
                print(f"Inner kwargs: {inner_kwargs}")
                
                # Execute the operator
                print("Executing operator with inner args/kwargs...")
                result = agent_result(*inner_args, **inner_kwargs)
                
                # Verify
                print("Verifying result...")
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"VERIFICATION FAILED for {inner_path}: {msg}")
                    all_passed = False
                else:
                    print(f"VERIFICATION PASSED for {inner_path}")
                    
            except Exception as e:
                print(f"ERROR: Failed processing inner path {inner_path}: {e}")
                traceback.print_exc()
                all_passed = False
        
        if all_passed:
            print("\nTEST PASSED")
            sys.exit(0)
        else:
            print("\nTEST FAILED")
            sys.exit(1)
    
    else:
        # Scenario A: Simple function
        print("Scenario A detected: Simple function call")
        
        result = agent_result
        expected = outer_output
        
        # Verify
        try:
            print("Verifying result...")
            passed, msg = recursive_check(expected, result)
            
            if passed:
                print("TEST PASSED")
                sys.exit(0)
            else:
                print(f"VERIFICATION FAILED: {msg}")
                sys.exit(1)
                
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()