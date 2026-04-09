import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_fft2c_torch import fft2c_torch
from verification_utils import recursive_check


def main():
    """Main test function for fft2c_torch"""
    
    # Data paths provided
    data_paths = ['/home/yjh/dpi_task2_sandbox/run_code/std_data/standard_data_fft2c_torch.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename:
            inner_paths.append(path)
        elif filename == 'standard_data_fft2c_torch.pkl':
            outer_path = path
    
    # Verify outer path exists
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_fft2c_torch.pkl)")
        sys.exit(1)
    
    if not os.path.exists(outer_path):
        print(f"ERROR: Outer data file does not exist: {outer_path}")
        sys.exit(1)
    
    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Successfully loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output')
    
    print(f"Outer data function name: {outer_data.get('func_name')}")
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Check if there are inner paths (Scenario B: Factory/Closure Pattern)
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print("\nDetected Factory/Closure Pattern (Scenario B)")
        
        # Phase 1: Create the operator
        try:
            agent_operator = fft2c_torch(*outer_args, **outer_kwargs)
            print("Successfully created operator from outer data")
        except Exception as e:
            print(f"ERROR: Failed to create operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify operator is callable
        if not callable(agent_operator):
            print(f"ERROR: Created operator is not callable, got type: {type(agent_operator)}")
            sys.exit(1)
        
        # Phase 2: Execute with inner data
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
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            print(f"Inner data function name: {inner_data.get('func_name')}")
            print(f"Inner args count: {len(inner_args)}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute the operator with inner args
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed operator with inner data")
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected, result)
                if passed:
                    print(f"TEST PASSED for inner data: {os.path.basename(inner_path)}")
                else:
                    print(f"TEST FAILED for inner data: {os.path.basename(inner_path)}")
                    print(f"Failure message: {msg}")
                    sys.exit(1)
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("\nTEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple Function
        print("\nDetected Simple Function Pattern (Scenario A)")
        
        # Execute the function directly
        try:
            result = fft2c_torch(*outer_args, **outer_kwargs)
            print("Successfully executed fft2c_torch")
        except Exception as e:
            print(f"ERROR: Failed to execute fft2c_torch: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected = outer_output
        
        # Debug info about result and expected
        if isinstance(result, torch.Tensor):
            print(f"Result type: torch.Tensor, shape: {result.shape}, dtype: {result.dtype}")
        else:
            print(f"Result type: {type(result)}")
        
        if isinstance(expected, torch.Tensor):
            print(f"Expected type: torch.Tensor, shape: {expected.shape}, dtype: {expected.dtype}")
        else:
            print(f"Expected type: {type(expected)}")
        
        # Compare results
        try:
            passed, msg = recursive_check(expected, result)
            if passed:
                print("\nTEST PASSED")
                sys.exit(0)
            else:
                print(f"\nTEST FAILED")
                print(f"Failure message: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()