import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_create_concentration_profiles import create_concentration_profiles
from verification_utils import recursive_check

def main():
    """Main test function for create_concentration_profiles."""
    
    # Define data paths
    data_paths = ['/data/yjh/spectrochempy_mcr_sandbox_sandbox/run_code/std_data/standard_data_create_concentration_profiles.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_create_concentration_profiles.pkl':
            outer_path = path
    
    # Validate outer path exists
    if outer_path is None:
        print("ERROR: Could not find outer data file (standard_data_create_concentration_profiles.pkl)")
        sys.exit(1)
    
    if not os.path.exists(outer_path):
        print(f"ERROR: Outer data file does not exist: {outer_path}")
        sys.exit(1)
    
    # Phase 1: Load outer data and execute function
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer args: {outer_args}")
        print(f"Outer kwargs: {outer_kwargs}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the function
    try:
        print("Executing create_concentration_profiles...")
        result = create_concentration_profiles(*outer_args, **outer_kwargs)
        print(f"Function executed successfully. Result type: {type(result)}")
        
    except Exception as e:
        print(f"ERROR: Failed to execute create_concentration_profiles: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Check if this is a factory pattern (result is callable and inner paths exist)
    inner_path = None
    for ip in inner_paths:
        if os.path.exists(ip):
            inner_path = ip
            break
    
    if inner_path is not None and callable(result):
        # Scenario B: Factory/Closure Pattern
        print(f"Detected factory pattern. Loading inner data from: {inner_path}")
        
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            print(f"Inner args: {inner_args}")
            print(f"Inner kwargs: {inner_kwargs}")
            
            # Execute the operator/closure
            print("Executing the returned operator...")
            actual_result = result(*inner_args, **inner_kwargs)
            
        except Exception as e:
            print(f"ERROR: Failed in factory pattern execution: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Scenario A: Simple function
        print("Using simple function pattern (no inner data or result not callable)")
        actual_result = result
        expected = outer_output
    
    # Phase 2: Verification
    try:
        print("Verifying results...")
        print(f"Expected type: {type(expected)}")
        print(f"Actual type: {type(actual_result)}")
        
        if isinstance(expected, np.ndarray):
            print(f"Expected shape: {expected.shape}, dtype: {expected.dtype}")
        if isinstance(actual_result, np.ndarray):
            print(f"Actual shape: {actual_result.shape}, dtype: {actual_result.dtype}")
        
        passed, msg = recursive_check(expected, actual_result)
        
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