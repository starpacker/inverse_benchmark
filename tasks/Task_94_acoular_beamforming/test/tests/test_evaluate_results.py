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
    data_paths = ['/data/yjh/acoular_beamforming_sandbox_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_evaluate_results.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute the function
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
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the function
    print("Executing evaluate_results with outer data...")
    try:
        result = evaluate_results(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if we have inner paths (factory/closure pattern)
    if inner_paths:
        # Scenario B: Factory pattern
        print(f"\nScenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # Verify result is callable
        if not callable(result):
            print(f"ERROR: Expected callable result for factory pattern, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        # Process each inner path
        for inner_path in inner_paths:
            print(f"\nProcessing inner data from: {inner_path}")
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
            
            print(f"Inner args count: {len(inner_args)}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute the operator
            print("Executing agent_operator with inner data...")
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            print("Comparing results...")
            try:
                passed, msg = recursive_check(expected, actual_result)
            except Exception as e:
                print(f"ERROR: Failed during comparison: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            
            print(f"Inner test passed: {msg}")
    else:
        # Scenario A: Simple function
        print("\nScenario A detected: Simple function execution")
        
        expected = outer_output
        
        # Compare results
        print("Comparing results...")
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: Failed during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    
    print("\nTEST PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()