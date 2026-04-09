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


def find_data_files(data_paths):
    """
    Analyze data paths to determine test scenario.
    Returns (outer_path, inner_path) where inner_path may be None.
    """
    outer_path = None
    inner_path = None
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_path = path
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
        elif 'forward_operator' in basename:
            # Fallback for outer path
            if outer_path is None:
                outer_path = path
    
    return outer_path, inner_path


def load_data(file_path):
    """Load data from pickle file using dill."""
    try:
        with open(file_path, 'rb') as f:
            data = dill.load(f)
        return data
    except Exception as e:
        print(f"FAILED: Error loading data from {file_path}: {e}")
        traceback.print_exc()
        sys.exit(1)


def main():
    # Define data paths
    data_paths = ['/home/yjh/diff_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Analyze data files to determine test scenario
    outer_path, inner_path = find_data_files(data_paths)
    
    if outer_path is None:
        print("FAILED: Could not find outer data file (standard_data_forward_operator.pkl)")
        sys.exit(1)
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data path: {inner_path}")
    
    # Phase 1: Load outer data and execute forward_operator
    print("\n=== Phase 1: Loading outer data and executing forward_operator ===")
    outer_data = load_data(outer_path)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    try:
        # Execute the forward_operator function
        result = forward_operator(*outer_args, **outer_kwargs)
        print("forward_operator executed successfully")
    except Exception as e:
        print(f"FAILED: Error executing forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine test scenario and verify
    if inner_path is not None:
        # Scenario B: Factory/Closure Pattern
        print("\n=== Phase 2 (Scenario B): Loading inner data and executing operator ===")
        
        # Verify that result is callable (it's an operator/closure)
        if not callable(result):
            print(f"FAILED: Expected callable operator from forward_operator, got {type(result)}")
            sys.exit(1)
        
        agent_operator = result
        print("Agent operator is callable")
        
        # Load inner data
        inner_data = load_data(inner_path)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected = inner_data.get('output', None)
        
        print(f"Inner args count: {len(inner_args)}")
        print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
        
        try:
            # Execute the operator with inner data
            actual_result = agent_operator(*inner_args, **inner_kwargs)
            print("Agent operator executed successfully")
        except Exception as e:
            print(f"FAILED: Error executing agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        result = actual_result
    else:
        # Scenario A: Simple Function
        print("\n=== Phase 2 (Scenario A): Simple function verification ===")
        expected = outer_output
        print("Using outer data output as expected result")
    
    # Phase 3: Comparison
    print("\n=== Phase 3: Verification ===")
    
    if expected is None:
        print("WARNING: Expected output is None, skipping comparison")
        print("TEST PASSED")
        sys.exit(0)
    
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"FAILED: Error during recursive_check: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"FAILED: Verification failed")
        print(f"Message: {msg}")
        
        # Additional debug info
        print(f"\nExpected type: {type(expected)}")
        print(f"Result type: {type(result)}")
        
        if isinstance(expected, torch.Tensor) and isinstance(result, torch.Tensor):
            print(f"Expected shape: {expected.shape}")
            print(f"Result shape: {result.shape}")
            print(f"Expected dtype: {expected.dtype}")
            print(f"Result dtype: {result.dtype}")
            
            if expected.shape == result.shape:
                diff = torch.abs(expected - result)
                print(f"Max absolute difference: {diff.max().item()}")
                print(f"Mean absolute difference: {diff.mean().item()}")
        
        sys.exit(1)


if __name__ == "__main__":
    main()