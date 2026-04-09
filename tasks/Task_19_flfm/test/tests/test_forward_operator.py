import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_forward_operator import forward_operator

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for forward_operator."""
    
    # Data paths provided
    data_paths = ['/home/yjh/flfm_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Analyze paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
    
    # Ensure we have the outer path
    if outer_path is None:
        print("ERROR: Could not find standard_data_forward_operator.pkl")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the forward_operator function
    try:
        print("Executing forward_operator...")
        result = forward_operator(*outer_args, **outer_kwargs)
        print(f"Execution completed. Result type: {type(result)}")
        
    except Exception as e:
        print(f"ERROR executing forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is Scenario B (factory/closure pattern)
    if inner_paths and callable(result):
        # Scenario B: Result is an operator/closure that needs to be called with inner data
        print("Detected Scenario B: Factory/Closure pattern")
        agent_operator = result
        
        for inner_path in inner_paths:
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output')
                
                print(f"Inner args count: {len(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator with inner data
                print("Executing agent_operator with inner data...")
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Inner execution completed. Result type: {type(result)}")
                
            except Exception as e:
                print(f"ERROR in Scenario B execution: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function - result is already the output
        print("Detected Scenario A: Simple function")
    
    # Phase 2: Verification
    try:
        print("Verifying results...")
        print(f"Expected type: {type(expected_output)}")
        print(f"Result type: {type(result)}")
        
        if isinstance(expected_output, torch.Tensor):
            print(f"Expected shape: {expected_output.shape}")
            print(f"Expected dtype: {expected_output.dtype}")
        if isinstance(result, torch.Tensor):
            print(f"Result shape: {result.shape}")
            print(f"Result dtype: {result.dtype}")
        
        passed, msg = recursive_check(expected_output, result)
        
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