import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the target function
from agent_forward_operator import forward_operator

# Import verification utility
from verification_utils import recursive_check

# Define helper function that may be needed for dill to load properly
def torch_complex_matmul(x, F):
    """Complex matrix multiplication for DFT."""
    Fx_real = torch.matmul(x, F[:, :, 0])
    Fx_imag = torch.matmul(x, F[:, :, 1])
    return torch.cat([Fx_real.unsqueeze(1), Fx_imag.unsqueeze(1)], -2)

def main():
    # Data paths provided
    data_paths = ['/home/yjh/dpi_task1_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Analyze data paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
    
    # Scenario A: Simple function - only outer data exists
    # Scenario B: Factory/Closure - both outer and inner data exist
    
    if outer_path is None:
        print("ERROR: Could not find standard_data_forward_operator.pkl")
        sys.exit(1)
    
    try:
        # Phase 1: Load outer data and run the function
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)
        
        print(f"Outer data loaded successfully")
        print(f"  Function name: {outer_data.get('func_name', 'unknown')}")
        print(f"  Number of args: {len(outer_args)}")
        print(f"  Kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    try:
        # Execute the forward_operator function
        print("\nExecuting forward_operator with loaded arguments...")
        result = forward_operator(*outer_args, **outer_kwargs)
        print("Function executed successfully")
        
    except Exception as e:
        print(f"ERROR executing forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if there are inner paths (Scenario B - factory pattern)
    if len(inner_paths) > 0:
        print(f"\nDetected factory/closure pattern with {len(inner_paths)} inner data file(s)")
        
        # If the result is callable, we need to test it with inner data
        if callable(result):
            for inner_path in inner_paths:
                try:
                    print(f"\nLoading inner data from: {inner_path}")
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                    
                    inner_args = inner_data.get('args', ())
                    inner_kwargs = inner_data.get('kwargs', {})
                    inner_expected = inner_data.get('output', None)
                    
                    print(f"Executing returned operator with inner arguments...")
                    inner_result = result(*inner_args, **inner_kwargs)
                    
                    # Compare inner result with expected
                    passed, msg = recursive_check(inner_expected, inner_result)
                    if not passed:
                        print(f"VERIFICATION FAILED for inner execution: {msg}")
                        sys.exit(1)
                    else:
                        print(f"Inner execution verification PASSED")
                        
                except Exception as e:
                    print(f"ERROR processing inner data {inner_path}: {e}")
                    traceback.print_exc()
                    sys.exit(1)
        else:
            # Result is not callable, compare directly with outer expected output
            print("Result is not callable, comparing with outer expected output...")
            try:
                passed, msg = recursive_check(expected_output, result)
                if not passed:
                    print(f"VERIFICATION FAILED: {msg}")
                    sys.exit(1)
            except Exception as e:
                print(f"ERROR during verification: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function, compare result directly
        print("\nScenario A: Simple function execution")
        print("Comparing result with expected output...")
        
        try:
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"VERIFICATION FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR during verification: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("\n" + "="*50)
    print("TEST PASSED")
    print("="*50)
    sys.exit(0)

if __name__ == '__main__':
    main()