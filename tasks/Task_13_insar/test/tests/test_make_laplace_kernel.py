import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_make_laplace_kernel import make_laplace_kernel

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for make_laplace_kernel"""
    
    # Data paths provided
    data_paths = ['/home/yjh/insar_sandbox/run_code/std_data/standard_data_make_laplace_kernel.pkl']
    
    # Separate outer and inner paths based on naming convention
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_make_laplace_kernel.pkl':
            outer_path = path
    
    # Validate that we have the outer path
    if outer_path is None:
        print("ERROR: Could not find standard_data_make_laplace_kernel.pkl in data_paths")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator/result
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    try:
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"Outer args: {outer_args}")
        print(f"Outer kwargs: {outer_kwargs}")
    except Exception as e:
        print(f"ERROR: Failed to extract outer data components: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the target function with outer args/kwargs
    try:
        print("Executing make_laplace_kernel with outer args/kwargs...")
        agent_result = make_laplace_kernel(*outer_args, **outer_kwargs)
        print(f"Agent result type: {type(agent_result)}")
    except Exception as e:
        print(f"ERROR: Failed to execute make_laplace_kernel: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and verify
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print(f"Scenario B detected: Found {len(inner_paths)} inner data file(s)")
        
        # Verify agent_result is callable for closure pattern
        if not callable(agent_result):
            print(f"WARNING: agent_result is not callable (type: {type(agent_result)})")
            print("Proceeding with Scenario A logic instead...")
            # Fall back to Scenario A
            result = agent_result
            expected = outer_output
        else:
            # Process inner paths
            for inner_path in inner_paths:
                try:
                    print(f"Loading inner data from: {inner_path}")
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                except Exception as e:
                    print(f"ERROR: Failed to load inner data: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                try:
                    inner_args = inner_data.get('args', ())
                    inner_kwargs = inner_data.get('kwargs', {})
                    expected = inner_data.get('output', None)
                    
                    print(f"Inner args: {inner_args}")
                    print(f"Inner kwargs: {inner_kwargs}")
                except Exception as e:
                    print(f"ERROR: Failed to extract inner data components: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                try:
                    print("Executing agent_result (operator) with inner args/kwargs...")
                    result = agent_result(*inner_args, **inner_kwargs)
                    print(f"Result type: {type(result)}")
                except Exception as e:
                    print(f"ERROR: Failed to execute agent_result: {e}")
                    traceback.print_exc()
                    sys.exit(1)
                
                # Verify this inner path result
                try:
                    print("Verifying result against expected output...")
                    passed, msg = recursive_check(expected, result)
                    if not passed:
                        print(f"TEST FAILED for inner path {inner_path}: {msg}")
                        sys.exit(1)
                    else:
                        print(f"Inner path {inner_path} verification passed")
                except Exception as e:
                    print(f"ERROR: Verification failed with exception: {e}")
                    traceback.print_exc()
                    sys.exit(1)
            
            # All inner paths passed
            print("TEST PASSED")
            sys.exit(0)
    else:
        # Scenario A: Simple Function
        print("Scenario A detected: Simple function test")
        result = agent_result
        expected = outer_output
    
    # Final verification for Scenario A (or fallback)
    try:
        print("Verifying result against expected output...")
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