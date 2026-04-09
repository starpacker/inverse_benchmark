import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_compute_bpm_gradient_batched import compute_bpm_gradient_batched

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for compute_bpm_gradient_batched."""
    
    # Data paths provided
    data_paths = ['/home/yjh/bpm_sandbox/run_code/std_data/standard_data_compute_bpm_gradient_batched.pkl']
    
    # Determine test scenario by analyzing file paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_compute_bpm_gradient_batched.pkl':
            outer_path = path
    
    # Validate we have the outer path
    if outer_path is None:
        print("ERROR: Could not find standard_data_compute_bpm_gradient_batched.pkl")
        sys.exit(1)
    
    # Check if outer path exists
    if not os.path.exists(outer_path):
        print(f"ERROR: Data file not found: {outer_path}")
        sys.exit(1)
    
    try:
        # Phase 1: Load outer data
        print(f"Loading outer data from: {outer_path}")
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
    
    # Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("Detected Scenario B: Factory/Closure Pattern")
        
        try:
            # Create the operator/closure
            print("Creating operator from compute_bpm_gradient_batched...")
            agent_operator = compute_bpm_gradient_batched(*outer_args, **outer_kwargs)
            
            # Verify it's callable
            if not callable(agent_operator):
                print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
                sys.exit(1)
            
            print("Operator created successfully")
            
        except Exception as e:
            print(f"ERROR: Failed to create operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Process each inner data file
        for inner_path in inner_paths:
            if not os.path.exists(inner_path):
                print(f"ERROR: Inner data file not found: {inner_path}")
                sys.exit(1)
            
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                # Execute the operator with inner args
                print("Executing operator with inner arguments...")
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Operator execution completed")
                
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                # Verify results
                print("Verifying results...")
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                
                print(f"Verification passed for {os.path.basename(inner_path)}")
                
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    
    else:
        # Scenario A: Simple Function
        print("Detected Scenario A: Simple Function")
        
        try:
            # Execute the function directly
            print("Executing compute_bpm_gradient_batched...")
            result = compute_bpm_gradient_batched(*outer_args, **outer_kwargs)
            print("Function execution completed")
            
        except Exception as e:
            print(f"ERROR: Failed to execute function: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        try:
            # Verify results against outer output
            expected = outer_output
            print("Verifying results...")
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()