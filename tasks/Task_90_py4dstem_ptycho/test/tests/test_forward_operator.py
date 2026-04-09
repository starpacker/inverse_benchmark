import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_forward_operator import forward_operator
from verification_utils import recursive_check

def main():
    """
    Test script for forward_operator function.
    
    This function computes diffraction pattern intensity from an object patch and probe.
    It's a simple function (not a factory/closure pattern).
    """
    
    # Define data paths
    data_paths = ['/data/yjh/py4dstem_ptycho_sandbox_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Analyze data paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_forward_operator.pkl':
            outer_path = path
    
    # Verify we have the main data file
    if outer_path is None:
        print("ERROR: Could not find standard_data_forward_operator.pkl")
        sys.exit(1)
    
    if not os.path.exists(outer_path):
        print(f"ERROR: Data file does not exist: {outer_path}")
        sys.exit(1)
    
    print(f"Loading data from: {outer_path}")
    
    # Load the outer/main data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load data file: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args, kwargs, and expected output
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output')
    
    print(f"Function name: {outer_data.get('func_name', 'forward_operator')}")
    print(f"Number of args: {len(outer_args)}")
    print(f"Number of kwargs: {len(outer_kwargs)}")
    
    # Check if there are inner paths (factory/closure pattern)
    valid_inner_paths = [p for p in inner_paths if os.path.exists(p)]
    
    if len(valid_inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("Detected factory/closure pattern with inner data files")
        
        # Phase 1: Create the operator
        try:
            print("Phase 1: Creating operator from forward_operator...")
            agent_operator = forward_operator(*outer_args, **outer_kwargs)
            
            if not callable(agent_operator):
                print(f"ERROR: Result is not callable. Got type: {type(agent_operator)}")
                sys.exit(1)
            
            print("Operator created successfully")
        except Exception as e:
            print(f"ERROR: Failed to create operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Phase 2: Execute with inner data and verify
        for inner_path in valid_inner_paths:
            print(f"\nProcessing inner data: {inner_path}")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            try:
                print("Executing operator with inner data...")
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verify result
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"VERIFICATION FAILED: {msg}")
                    sys.exit(1)
                print(f"Verification passed for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("\nTEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple Function
        print("Detected simple function pattern")
        
        # Execute the function
        try:
            print("Executing forward_operator...")
            result = forward_operator(*outer_args, **outer_kwargs)
            print(f"Execution successful. Result type: {type(result)}")
            
            if isinstance(result, np.ndarray):
                print(f"Result shape: {result.shape}, dtype: {result.dtype}")
        except Exception as e:
            print(f"ERROR: Failed to execute forward_operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify result
        try:
            print("Verifying result against expected output...")
            passed, msg = recursive_check(expected_output, result)
            
            if not passed:
                print(f"VERIFICATION FAILED: {msg}")
                
                # Additional debugging info
                if isinstance(expected_output, np.ndarray) and isinstance(result, np.ndarray):
                    print(f"Expected shape: {expected_output.shape}, dtype: {expected_output.dtype}")
                    print(f"Result shape: {result.shape}, dtype: {result.dtype}")
                    if expected_output.shape == result.shape:
                        diff = np.abs(expected_output - result)
                        print(f"Max absolute difference: {np.max(diff)}")
                        print(f"Mean absolute difference: {np.mean(diff)}")
                
                sys.exit(1)
            
            print("Verification passed")
            print("\nTEST PASSED")
            sys.exit(0)
            
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()