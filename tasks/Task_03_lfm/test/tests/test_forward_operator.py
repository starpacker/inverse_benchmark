import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add path for imports if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_forward_operator import forward_operator
from verification_utils import recursive_check

def main():
    """Main test function for forward_operator."""
    
    data_paths = ['/home/yjh/lfm_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
    
    # Verify outer path exists
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_forward_operator.pkl)")
        sys.exit(1)
    
    if not os.path.exists(outer_path):
        print(f"ERROR: Outer data file does not exist: {outer_path}")
        sys.exit(1)
    
    # Phase 1: Load outer data and run forward_operator
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer data loaded successfully.")
        print(f"  - Number of args: {len(outer_args)}")
        print(f"  - Kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute forward_operator with outer data
    try:
        print("Executing forward_operator with outer data...")
        result = forward_operator(*outer_args, **outer_kwargs)
        print(f"forward_operator executed successfully.")
        print(f"  - Result type: {type(result)}")
        
    except Exception as e:
        print(f"ERROR: Failed to execute forward_operator: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if result is callable (factory pattern) and inner paths exist
    if callable(result) and not isinstance(result, (np.ndarray, type)):
        # Scenario B: Factory/Closure pattern
        agent_operator = result
        print("Result is callable - detected factory/closure pattern.")
        
        # Look for inner data files
        if inner_paths:
            inner_path = inner_paths[0]  # Use first inner path
            
            if not os.path.exists(inner_path):
                print(f"ERROR: Inner data file does not exist: {inner_path}")
                sys.exit(1)
            
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Inner data loaded successfully.")
                print(f"  - Number of args: {len(inner_args)}")
                print(f"  - Kwargs keys: {list(inner_kwargs.keys())}")
                
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Execute the returned operator with inner data
            try:
                print("Executing returned operator with inner data...")
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Operator executed successfully.")
                print(f"  - Result type: {type(result)}")
                
            except Exception as e:
                print(f"ERROR: Failed to execute returned operator: {e}")
                traceback.print_exc()
                sys.exit(1)
        else:
            # No inner path but result is callable - compare the callable itself
            print("No inner data found. Comparing callable result with expected output.")
            expected = outer_output
    else:
        # Scenario A: Simple function - result is the direct output
        print("Result is not callable - simple function pattern.")
        expected = outer_output
    
    # Phase 2: Verification
    try:
        print("Performing verification...")
        print(f"  - Expected type: {type(expected)}")
        print(f"  - Result type: {type(result)}")
        
        # Handle cupy arrays if present
        try:
            import cupy
            if isinstance(result, cupy.ndarray):
                result = cupy.asnumpy(result)
                print("  - Converted result from cupy to numpy")
            if isinstance(expected, cupy.ndarray):
                expected = cupy.asnumpy(expected)
                print("  - Converted expected from cupy to numpy")
        except ImportError:
            pass
        
        passed, msg = recursive_check(expected, result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()