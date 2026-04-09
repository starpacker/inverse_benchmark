import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_visualize_results_internal import visualize_results_internal
from verification_utils import recursive_check

def main():
    """Main test function for visualize_results_internal"""
    
    # Data paths provided
    data_paths = ['/data/yjh/aspire_cryoem_sandbox_sandbox/run_code/std_data/standard_data_visualize_results_internal.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_visualize_results_internal.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_visualize_results_internal.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
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
    
    print(f"Outer data loaded. Function: {outer_data.get('func_name', 'unknown')}")
    print(f"Args count: {len(outer_args)}, Kwargs keys: {list(outer_kwargs.keys())}")
    
    # Execute the function
    try:
        print("Executing visualize_results_internal...")
        result = visualize_results_internal(*outer_args, **outer_kwargs)
        print("Function executed successfully.")
    except Exception as e:
        print(f"ERROR: Failed to execute visualize_results_internal: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if we have inner paths (factory/closure pattern)
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print(f"\nScenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # Verify the result is callable
        if not callable(result):
            print(f"ERROR: Expected callable operator, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        
        for inner_path in inner_paths:
            print(f"\nProcessing inner data: {inner_path}")
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
            
            print(f"Inner data loaded. Function: {inner_data.get('func_name', 'unknown')}")
            
            # Execute the operator with inner args
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            print("Comparing results...")
            passed, msg = recursive_check(expected, actual_result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print(f"Inner test passed: {msg}")
        
        print("\nTEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple function
        print("\nScenario A detected: Simple function call")
        
        expected = outer_output
        
        # Compare results
        print("Comparing results...")
        passed, msg = recursive_check(expected, result)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print(f"TEST PASSED: {msg}")
            sys.exit(0)

if __name__ == "__main__":
    main()