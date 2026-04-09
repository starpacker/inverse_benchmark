import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_evaluate_results import evaluate_results
from verification_utils import recursive_check


def main():
    """Main test function for evaluate_results."""
    
    data_paths = ['/home/yjh/lensless_admm_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Separate outer and inner paths
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
    
    # Phase 1: Load outer data and call evaluate_results
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
    
    # Execute the function
    try:
        print("Executing evaluate_results with outer args/kwargs...")
        result = evaluate_results(*outer_args, **outer_kwargs)
        print(f"Function executed successfully. Result type: {type(result)}")
        
    except Exception as e:
        print(f"ERROR executing evaluate_results: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario based on inner paths
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print(f"\nScenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # Check if result is callable (operator/closure)
        if not callable(result):
            print(f"WARNING: Result is not callable but inner paths exist. Type: {type(result)}")
            # Fall back to comparing with outer output
            expected = outer_output
        else:
            # Process inner data files
            agent_operator = result
            
            for inner_path in inner_paths:
                try:
                    print(f"\nLoading inner data from: {inner_path}")
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                    
                    inner_args = inner_data.get('args', ())
                    inner_kwargs = inner_data.get('kwargs', {})
                    expected = inner_data.get('output', None)
                    
                    print(f"Inner args count: {len(inner_args)}")
                    print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                    
                    print("Executing agent_operator with inner args/kwargs...")
                    result = agent_operator(*inner_args, **inner_kwargs)
                    print(f"Operator executed successfully. Result type: {type(result)}")
                    
                except Exception as e:
                    print(f"ERROR processing inner data: {e}")
                    traceback.print_exc()
                    sys.exit(1)
    else:
        # Scenario A: Simple Function
        print("\nScenario A detected: Simple function call")
        expected = outer_output
    
    # Phase 2: Verification
    try:
        print("\nPerforming verification...")
        print(f"Expected type: {type(expected)}")
        print(f"Result type: {type(result)}")
        
        passed, msg = recursive_check(expected, result)
        
        if passed:
            print("\n" + "="*50)
            print("TEST PASSED")
            print("="*50)
            sys.exit(0)
        else:
            print("\n" + "="*50)
            print("TEST FAILED")
            print(f"Verification message: {msg}")
            print("="*50)
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during verification: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()