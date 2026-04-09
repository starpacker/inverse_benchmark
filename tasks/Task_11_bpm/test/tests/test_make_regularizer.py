import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_make_regularizer import make_regularizer
from verification_utils import recursive_check


def find_data_files(data_paths):
    """Separate outer and inner data files."""
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_make_regularizer.pkl':
            outer_path = path
    
    return outer_path, inner_paths


def test_make_regularizer():
    """Test the make_regularizer function."""
    
    data_paths = ['/home/yjh/bpm_sandbox/run_code/std_data/standard_data_make_regularizer.pkl']
    
    # Find data files
    outer_path, inner_paths = find_data_files(data_paths)
    
    if outer_path is None:
        print("FAILED: Could not find outer data file")
        sys.exit(1)
    
    print(f"Found outer data file: {outer_path}")
    print(f"Found {len(inner_paths)} inner data file(s)")
    
    # === Phase 1: Reconstruct Operator ===
    print("\n=== Phase 1: Reconstructing Operator ===")
    
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"FAILED: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    try:
        agent_operator = make_regularizer(*outer_args, **outer_kwargs)
        print(f"Operator created successfully: {type(agent_operator)}")
    except Exception as e:
        print(f"FAILED: Could not create operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Verify operator is callable
    if not callable(agent_operator):
        print(f"FAILED: Operator is not callable, got {type(agent_operator)}")
        sys.exit(1)
    
    # === Phase 2: Execution & Verification ===
    print("\n=== Phase 2: Execution & Verification ===")
    
    if len(inner_paths) > 0:
        # Scenario B: Test with inner data
        print("Scenario B: Testing with inner data (closure execution)")
        
        for inner_path in inner_paths:
            print(f"\nTesting with: {os.path.basename(inner_path)}")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAILED: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAILED: Operator execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAILED: Verification failed")
                print(f"Message: {msg}")
                sys.exit(1)
            
            print(f"Inner test passed")
    else:
        # Scenario A: No inner data - test the operator behavior
        print("Scenario A: Testing operator behavior (factory pattern)")
        
        # Since the expected output is a function, we can't compare functions directly.
        # Instead, we verify the operator works correctly by testing its behavior.
        
        # Extract ROI and device from kwargs to create appropriate test input
        ROI = outer_kwargs.get('ROI', outer_args[3] if len(outer_args) > 3 else None)
        device = outer_kwargs.get('device', outer_args[5] if len(outer_args) > 5 else 'cpu')
        
        # Create a test tensor based on ROI dimensions
        if ROI is not None:
            s0, e0, s1, e1, s2, e2 = ROI
            # Create tensor slightly larger than ROI
            test_shape = (max(e0 + 2, 10), max(e1 + 2, 10), max(e2 + 2, 10))
        else:
            test_shape = (10, 10, 10)
        
        try:
            # Create test input tensor
            test_input = torch.randn(test_shape, dtype=torch.float32, device=device)
            
            # Execute the operator
            result = agent_operator(test_input.clone())
            
            print(f"Test input shape: {test_input.shape}")
            print(f"Result shape: {result.shape}")
            print(f"Result type: {type(result)}")
            
            # Verify result is a tensor with same shape
            if not isinstance(result, torch.Tensor):
                print(f"FAILED: Expected torch.Tensor, got {type(result)}")
                sys.exit(1)
            
            if result.shape != test_input.shape:
                print(f"FAILED: Shape mismatch - expected {test_input.shape}, got {result.shape}")
                sys.exit(1)
            
            # Verify the operator is deterministic (same input gives same output)
            test_input_copy = test_input.clone()
            result2 = agent_operator(test_input_copy)
            
            passed, msg = recursive_check(result, result2)
            if not passed:
                print(f"FAILED: Operator is not deterministic")
                print(f"Message: {msg}")
                sys.exit(1)
            
            print("Operator behavior verified successfully")
            
        except Exception as e:
            print(f"FAILED: Operator execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    test_make_regularizer()