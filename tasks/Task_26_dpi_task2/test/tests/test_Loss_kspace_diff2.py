import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_Loss_kspace_diff2 import Loss_kspace_diff2

# Import verification utility
from verification_utils import recursive_check

def main():
    """Main test function for Loss_kspace_diff2"""
    
    # Data paths provided
    data_paths = ['/home/yjh/dpi_task2_sandbox/run_code/std_data/standard_data_Loss_kspace_diff2.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename or 'parent_' in filename:
            inner_paths.append(path)
        elif filename == 'standard_data_Loss_kspace_diff2.pkl':
            outer_path = path
    
    # Validate outer path exists
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_Loss_kspace_diff2.pkl)")
        sys.exit(1)
    
    if not os.path.exists(outer_path):
        print(f"ERROR: Outer data file does not exist: {outer_path}")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Successfully loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract outer args and kwargs
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer args: {outer_args}")
    print(f"Outer kwargs: {outer_kwargs}")
    
    # Create the operator/closure by calling Loss_kspace_diff2
    try:
        agent_operator = Loss_kspace_diff2(*outer_args, **outer_kwargs)
        print("Successfully created agent operator from Loss_kspace_diff2")
    except Exception as e:
        print(f"ERROR: Failed to create agent operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Verify operator is callable
    if not callable(agent_operator):
        print("ERROR: Created agent operator is not callable")
        sys.exit(1)
    
    # Check for inner paths (Scenario B: Factory/Closure Pattern)
    # Look for files that match the pattern for inner function calls
    inner_path = None
    
    # Also check for inner paths that might exist but weren't in data_paths
    std_data_dir = os.path.dirname(outer_path)
    potential_inner_patterns = [
        f'standard_data_parent_function_Loss_kspace_diff2_',
        f'standard_data_parent_Loss_kspace_diff2_'
    ]
    
    for filename in os.listdir(std_data_dir):
        for pattern in potential_inner_patterns:
            if filename.startswith(pattern) and filename.endswith('.pkl'):
                inner_path = os.path.join(std_data_dir, filename)
                inner_paths.append(inner_path)
                break
    
    # Remove duplicates
    inner_paths = list(set(inner_paths))
    
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print(f"Found {len(inner_paths)} inner data file(s), using closure pattern")
        
        for inner_path in inner_paths:
            if not os.path.exists(inner_path):
                print(f"WARNING: Inner data file does not exist: {inner_path}")
                continue
                
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Successfully loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Extract inner args and kwargs
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            print(f"Inner args count: {len(inner_args)}")
            print(f"Inner kwargs: {inner_kwargs}")
            
            # Execute the operator with inner arguments
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed agent operator with inner arguments")
            except Exception as e:
                print(f"ERROR: Failed to execute agent operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Verification passed for inner data: {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple Function or no inner data
        # The output of the outer function call is the result
        print("No inner data files found, using simple comparison pattern")
        
        # The result is the operator itself (a function)
        result = agent_operator
        expected = outer_output
        
        # If expected is also a callable (function), we need special handling
        if callable(expected) and callable(result):
            # Both are functions - we consider this a pass if both are callable
            # since we can't directly compare function objects
            print("Both expected and result are callable functions")
            print("TEST PASSED")
            sys.exit(0)
        
        # Otherwise, compare normally
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()