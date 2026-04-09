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
    
    # Data paths provided
    data_paths = ['/home/yjh/lensless_dl_sandbox/run_code/std_data/standard_data_evaluate_results.pkl']
    
    # Separate outer and inner paths based on naming convention
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_evaluate_results.pkl':
            outer_path = path
    
    # Verify we have at least the outer path
    if outer_path is None:
        print("ERROR: Could not find standard_data_evaluate_results.pkl in data_paths")
        sys.exit(1)
    
    print(f"Outer path: {outer_path}")
    print(f"Inner paths: {inner_paths}")
    
    # Phase 1: Load outer data and execute function
    try:
        print("\n=== Phase 1: Loading outer data ===")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine scenario based on presence of inner paths
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("\n=== Scenario B: Factory/Closure Pattern ===")
        
        try:
            print("\n=== Creating operator from outer data ===")
            agent_operator = evaluate_results(*outer_args, **outer_kwargs)
            
            # Verify the operator is callable
            if not callable(agent_operator):
                print(f"WARNING: agent_operator is not callable, type: {type(agent_operator)}")
                # In some cases, the function might return None or a non-callable
                # We'll proceed with inner data execution if possible
            
        except Exception as e:
            print(f"ERROR: Failed to create operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Process each inner path
        for inner_path in inner_paths:
            try:
                print(f"\n=== Processing inner data: {inner_path} ===")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner data
                print("\n=== Executing operator with inner data ===")
                result = agent_operator(*inner_args, **inner_kwargs)
                
                # Compare results
                print("\n=== Comparing results ===")
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed: {inner_path}")
                    
            except Exception as e:
                print(f"ERROR: Failed processing inner data {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("\nTEST PASSED")
        sys.exit(0)
        
    else:
        # Scenario A: Simple Function
        print("\n=== Scenario A: Simple Function ===")
        
        try:
            print("\n=== Executing evaluate_results ===")
            result = evaluate_results(*outer_args, **outer_kwargs)
            
            expected = outer_output
            
            # Compare results
            print("\n=== Comparing results ===")
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
                
        except Exception as e:
            print(f"ERROR: Failed to execute evaluate_results: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()