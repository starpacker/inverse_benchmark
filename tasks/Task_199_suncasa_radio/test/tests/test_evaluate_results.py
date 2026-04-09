import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check

def main():
    """Main test function for evaluate_results."""
    
    # Data paths provided
    data_paths = ['/data/yjh/suncasa_radio_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = p
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_evaluate_results.pkl)")
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
    outer_output = outer_data.get('output')
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the function
    try:
        print("Executing evaluate_results with outer args/kwargs...")
        result = evaluate_results(*outer_args, **outer_kwargs)
        print("Function executed successfully.")
    except Exception as e:
        print(f"ERROR: Function execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if result is callable (factory pattern)
    if callable(result) and not isinstance(result, type) and len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected factory pattern. Result is callable.")
        agent_operator = result
        
        # Load inner data
        inner_path = inner_paths[0]
        print(f"Loading inner data from: {inner_path}")
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected = inner_data.get('output')
        
        print(f"Inner args count: {len(inner_args)}")
        print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
        
        # Execute the operator
        try:
            print("Executing agent_operator with inner args/kwargs...")
            actual_result = agent_operator(*inner_args, **inner_kwargs)
            print("Operator executed successfully.")
        except Exception as e:
            print(f"ERROR: Operator execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        result = actual_result
    else:
        # Scenario A: Simple function
        print("Detected simple function pattern.")
        expected = outer_output
    
    # Phase 2: Verification
    print("Verifying results...")
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()