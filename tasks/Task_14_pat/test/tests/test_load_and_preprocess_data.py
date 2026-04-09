import sys
import os
import dill
import numpy as np
import traceback

# Add the current directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check

def main():
    """Main test function for load_and_preprocess_data."""
    
    # Data paths provided
    data_paths = ['/home/yjh/pat_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # File Logic Setup: Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    # Sort inner paths for consistent ordering
    inner_paths.sort()
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_load_and_preprocess_data.pkl)")
        sys.exit(1)
    
    print(f"Outer path: {outer_path}")
    print(f"Inner paths: {inner_paths}")
    
    try:
        # Phase 1: Load outer data and execute function
        print("\n=== Phase 1: Loading outer data ===")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"Outer args: {outer_args}")
        print(f"Outer kwargs: {outer_kwargs}")
        
        # Execute the function
        print("\n=== Executing load_and_preprocess_data ===")
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        
        # Determine scenario based on inner paths
        if inner_paths:
            # Scenario B: Factory/Closure Pattern
            print("\n=== Scenario B: Factory/Closure Pattern ===")
            
            # Verify the result is callable (an operator)
            if not callable(result):
                print(f"WARNING: Result is not callable, treating as Scenario A")
                # Fall back to Scenario A
                expected = outer_output
            else:
                agent_operator = result
                
                # Process each inner path
                for inner_path in inner_paths:
                    print(f"\n=== Processing inner data: {inner_path} ===")
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                    
                    inner_args = inner_data.get('args', ())
                    inner_kwargs = inner_data.get('kwargs', {})
                    expected = inner_data.get('output', None)
                    
                    print(f"Inner args: {inner_args}")
                    print(f"Inner kwargs: {inner_kwargs}")
                    
                    # Execute the operator with inner args
                    result = agent_operator(*inner_args, **inner_kwargs)
                    
                    # Verify
                    print("\n=== Verification ===")
                    passed, msg = recursive_check(expected, result)
                    
                    if not passed:
                        print(f"TEST FAILED: {msg}")
                        sys.exit(1)
                    else:
                        print(f"Inner test passed: {msg}")
                
                print("\nTEST PASSED")
                sys.exit(0)
        else:
            # Scenario A: Simple Function
            print("\n=== Scenario A: Simple Function ===")
            expected = outer_output
        
        # Verification for Scenario A (or fallback)
        print("\n=== Verification ===")
        passed, msg = recursive_check(expected, result)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print(f"TEST PASSED: {msg}")
            sys.exit(0)
            
    except Exception as e:
        print(f"ERROR during test execution: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()