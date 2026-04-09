import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_torch_complex_matmul import torch_complex_matmul
from verification_utils import recursive_check


def main():
    """Main test function for torch_complex_matmul."""
    
    # Data paths provided
    data_paths = ['/home/yjh/dpi_task1_sandbox/run_code/std_data/standard_data_torch_complex_matmul.pkl']
    
    # Analyze data files to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        basename = os.path.basename(path)
        
        # Check if this is an inner data file (contains 'parent_function' or 'parent_')
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_torch_complex_matmul.pkl':
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_torch_complex_matmul.pkl)")
        sys.exit(1)
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and execute the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print(f"Outer data keys: {outer_data.keys()}")
    print(f"Function name: {outer_data.get('func_name', 'N/A')}")
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    print(f"Number of outer args: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Determine if this is Scenario A (simple function) or Scenario B (factory/closure)
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("\n=== Scenario B: Factory/Closure Pattern Detected ===")
        
        # Execute the outer function to get the operator/closure
        try:
            agent_operator = torch_complex_matmul(*outer_args, **outer_kwargs)
            print(f"Agent operator created: {type(agent_operator)}")
        except Exception as e:
            print(f"ERROR: Failed to create agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify the operator is callable
        if not callable(agent_operator):
            print(f"ERROR: Agent operator is not callable, got type: {type(agent_operator)}")
            sys.exit(1)
        
        # Phase 2: Load inner data and execute the operator
        for inner_path in inner_paths:
            print(f"\n--- Testing with inner data: {inner_path} ---")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)
            
            print(f"Inner data keys: {inner_data.keys()}")
            print(f"Number of inner args: {len(inner_args)}")
            
            # Execute the operator with inner args
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(inner_expected, actual_result)
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print(f"Inner test passed: {msg}")
        
        print("\nTEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple Function
        print("\n=== Scenario A: Simple Function ===")
        
        # Execute the function
        try:
            actual_result = torch_complex_matmul(*outer_args, **outer_kwargs)
            print(f"Result type: {type(actual_result)}")
            if hasattr(actual_result, 'shape'):
                print(f"Result shape: {actual_result.shape}")
        except Exception as e:
            print(f"ERROR: Failed to execute torch_complex_matmul: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Compare with expected output
        try:
            passed, msg = recursive_check(expected_output, actual_result)
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print(f"TEST PASSED: {msg}")
            sys.exit(0)


if __name__ == '__main__':
    main()