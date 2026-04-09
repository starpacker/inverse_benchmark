import sys
import os
import dill
import numpy as np
import scipy.ndimage
import scipy.fftpack
import traceback
from numpy.fft import rfft2, irfft2

# Import the target function
from agent_forward_operator import forward_operator
from verification_utils import recursive_check

def unpad(img, npad):
    return img[npad:-npad, npad:-npad]

def _centered(arr, newshape):
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

def run_test():
    # Configuration
    data_paths = ['/data/yjh/semiblindpsfdeconv-master_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Identify Data Files
    outer_path = None
    inner_path = None
    
    for path in data_paths:
        if 'standard_data_forward_operator.pkl' in path:
            outer_path = path
        elif 'standard_data_parent_function_forward_operator' in path:
            inner_path = path
            
    if not outer_path:
        print("CRITICAL: Outer data file 'standard_data_forward_operator.pkl' not found.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Scenario Detection
    # -------------------------------------------------------------------------
    # If inner_path exists, it implies `forward_operator` returned a callable (closure/factory pattern).
    # If inner_path does NOT exist, `forward_operator` returned the final result directly.
    # Based on the provided code, `forward_operator` returns `blurred_image` (a numpy array),
    # so we expect Scenario A (Simple Function).
    
    try:
        # Load Outer Data
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        print(f"Loaded outer data from: {outer_path}")
        
        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        # Execute Function
        print("Executing forward_operator with loaded arguments...")
        actual_output = forward_operator(*args, **kwargs)
        
        # In this specific codebase analysis, `forward_operator` returns a value, not a function.
        # However, we add a check just in case the provided code was a simplified version 
        # but the data implies a factory pattern (which happens in decorators sometimes).
        
        if callable(actual_output) and not isinstance(actual_output, (np.ndarray, list, dict, tuple)):
            if inner_path:
                print("Detected Factory Pattern. Loading inner data...")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', [])
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output') # Update expected output to the inner result
                
                print("Executing generated operator with inner arguments...")
                actual_output = actual_output(*inner_args, **inner_kwargs)
            else:
                # If it returns a callable but we have no inner data, we can't fully test execution,
                # but we can check if the output type matches expected (which might be the function itself).
                pass

        # Verification
        print("Verifying results...")
        passed, msg = recursive_check(expected_output, actual_output)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"An error occurred during test execution:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()