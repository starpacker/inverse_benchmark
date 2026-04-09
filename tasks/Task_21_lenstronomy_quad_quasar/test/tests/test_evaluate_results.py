import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_evaluate_results import evaluate_results

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for evaluate_results."""
    
    # Define data paths
    data_paths = ['/home/yjh/lenstronomy_quad_quasar_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Determine test scenario by analyzing data paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_evaluate_results.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the target function
    try:
        print("Executing evaluate_results with loaded arguments...")
        result = evaluate_results(*outer_args, **outer_kwargs)
        print("Function executed successfully.")
        
    except Exception as e:
        print(f"ERROR executing evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine which scenario we're in
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print(f"\nScenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # Check if result is callable (an operator/closure)
        if not callable(result):
            print("WARNING: Result is not callable, treating as Scenario A instead")
            expected = outer_output
        else:
            # Load inner data and execute the operator
            try:
                inner_path = inner_paths[0]  # Use first inner path
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner arguments
                print("Executing operator with inner arguments...")
                result = result(*inner_args, **inner_kwargs)
                print("Operator executed successfully.")
                
            except Exception as e:
                print(f"ERROR in Scenario B execution: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple Function
        print("\nScenario A detected: Simple function call")
        expected = outer_output
    
    # Phase 2: Verification
    try:
        print("\nVerifying results...")
        passed, msg = recursive_check(expected, result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during verification: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()