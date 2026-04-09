import sys
import os
import dill
import numpy as np
import traceback

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check


def main():
    """Main test function for evaluate_results."""
    
    data_paths = ['/home/yjh/oopao_zernike_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Categorize the data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file 'standard_data_evaluate_results.pkl'")
        sys.exit(1)
    
    # Phase 1: Load outer data and run the function
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer data loaded. Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"  Args count: {len(outer_args)}")
    print(f"  Kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the function with outer args/kwargs
    print("\nExecuting evaluate_results with outer arguments...")
    try:
        result = evaluate_results(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("Function executed successfully.")
    
    # Check if this is Scenario A or B
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"\nScenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # Check if result is callable (operator/closure)
        if callable(result):
            agent_operator = result
            print("Result is callable - proceeding with inner data execution")
            
            # Process each inner path
            for inner_path in inner_paths:
                print(f"\nLoading inner data from: {inner_path}")
                try:
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                except Exception as e:
                    print(f"ERROR: Failed to load inner data: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print(f"Inner data loaded. Function name: {inner_data.get('func_name', 'unknown')}")
                print(f"  Args count: {len(inner_args)}")
                print(f"  Kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner args/kwargs
                print("\nExecuting agent_operator with inner arguments...")
                try:
                    actual_result = agent_operator(*inner_args, **inner_kwargs)
                except Exception as e:
                    print(f"ERROR: Failed to execute agent_operator: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                # Compare results
                print("\nComparing results...")
                try:
                    passed, msg = recursive_check(expected, actual_result)
                except Exception as e:
                    print(f"ERROR: Verification failed with exception: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed: {msg}")
            
            print("\nTEST PASSED")
            sys.exit(0)
        else:
            # Result is not callable, but we have inner paths - this might be an error
            # or the inner paths are for a different purpose
            print("Result is not callable, falling back to Scenario A comparison")
            expected = outer_output
    else:
        # Scenario A: Simple function
        print("\nScenario A detected: Simple function test")
        expected = outer_output
    
    # Scenario A comparison
    print("\nComparing results against outer data output...")
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if not passed:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)
    else:
        print(f"TEST PASSED: {msg}")
        sys.exit(0)


if __name__ == '__main__':
    main()