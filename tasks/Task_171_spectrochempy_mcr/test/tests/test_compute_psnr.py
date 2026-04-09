import sys
import os
import dill
import traceback
import numpy as np

# Import the target function
from agent_compute_psnr import compute_psnr
from verification_utils import recursive_check

def main():
    """Main test function for compute_psnr."""
    
    # Data paths provided
    data_paths = ['/data/yjh/spectrochempy_mcr_sandbox_sandbox/run_code/std_data/standard_data_compute_psnr.pkl']
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(path)
        elif basename == 'standard_data_compute_psnr.pkl':
            outer_path = path
    
    # Verify we have the outer path
    if outer_path is None:
        print("ERROR: Could not find standard_data_compute_psnr.pkl")
        sys.exit(1)
    
    # Check if file exists
    if not os.path.exists(outer_path):
        print(f"ERROR: Data file does not exist: {outer_path}")
        sys.exit(1)
    
    try:
        # Phase 1: Load outer data and execute function
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Executing compute_psnr with {len(outer_args)} args and {len(outer_kwargs)} kwargs")
        
        # Execute the function
        result = compute_psnr(*outer_args, **outer_kwargs)
        
        # Determine if this is Scenario A or B
        if inner_paths:
            # Scenario B: Factory/Closure pattern
            print("Detected Scenario B: Factory/Closure pattern")
            
            # Verify result is callable
            if not callable(result):
                print(f"ERROR: Expected callable from compute_psnr, got {type(result)}")
                sys.exit(1)
            
            agent_operator = result
            
            # Process inner paths
            for inner_path in inner_paths:
                if not os.path.exists(inner_path):
                    print(f"WARNING: Inner data file does not exist: {inner_path}")
                    continue
                
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Executing operator with {len(inner_args)} args and {len(inner_kwargs)} kwargs")
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                
                # Compare results
                passed, msg = recursive_check(expected, actual_result)
                
                if not passed:
                    print(f"TEST FAILED for inner execution")
                    print(f"Failure message: {msg}")
                    sys.exit(1)
                
                print(f"Inner execution passed")
            
            print("TEST PASSED")
            sys.exit(0)
        
        else:
            # Scenario A: Simple function
            print("Detected Scenario A: Simple function")
            
            expected = outer_output
            
            # Compare results
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"TEST FAILED")
                print(f"Failure message: {msg}")
                print(f"Expected: {expected}")
                print(f"Actual: {result}")
                sys.exit(1)
            
            print("TEST PASSED")
            sys.exit(0)
    
    except Exception as e:
        print(f"ERROR: Exception during test execution")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()