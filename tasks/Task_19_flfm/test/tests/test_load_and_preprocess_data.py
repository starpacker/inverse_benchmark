import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_load_and_preprocess_data import load_and_preprocess_data

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for load_and_preprocess_data."""
    
    # Data paths provided
    data_paths = ['/home/yjh/flfm_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
    
    # Validate that we have the outer path
    if outer_path is None:
        print("ERROR: Could not find outer data file 'standard_data_load_and_preprocess_data.pkl'")
        sys.exit(1)
    
    # Sort inner paths for consistent ordering
    inner_paths.sort()
    
    try:
        # Phase 1: Load outer data and reconstruct operator/result
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"Outer args: {outer_args}")
        print(f"Outer kwargs: {outer_kwargs}")
        
        # Execute the function with outer arguments
        print("Executing load_and_preprocess_data with outer arguments...")
        agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        
        # Check if this is Scenario A (simple function) or Scenario B (factory/closure pattern)
        if len(inner_paths) > 0:
            # Scenario B: Factory/Closure Pattern
            print(f"Detected Scenario B: Factory/Closure Pattern with {len(inner_paths)} inner data file(s)")
            
            # Verify agent_result is callable
            if not callable(agent_result):
                print(f"ERROR: Expected callable operator, got {type(agent_result)}")
                sys.exit(1)
            
            agent_operator = agent_result
            
            # Process each inner data file
            for inner_path in inner_paths:
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output', None)
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner arguments
                print("Executing operator with inner arguments...")
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                
                # Verify results
                print("Verifying results...")
                passed, msg = recursive_check(expected_output, actual_result)
                
                if not passed:
                    print(f"TEST FAILED for {os.path.basename(inner_path)}")
                    print(f"Verification message: {msg}")
                    sys.exit(1)
                else:
                    print(f"Verification passed for {os.path.basename(inner_path)}")
            
            print("\nTEST PASSED")
            sys.exit(0)
        
        else:
            # Scenario A: Simple Function
            print("Detected Scenario A: Simple Function")
            
            # The result from Phase 1 IS the result to compare
            result = agent_result
            expected = outer_output
            
            # Verify results
            print("Verifying results...")
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print("TEST FAILED")
                print(f"Verification message: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
    
    except FileNotFoundError as e:
        print(f"ERROR: File not found - {e}")
        traceback.print_exc()
        sys.exit(1)
    
    except Exception as e:
        print(f"ERROR: Unexpected exception during test execution - {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()