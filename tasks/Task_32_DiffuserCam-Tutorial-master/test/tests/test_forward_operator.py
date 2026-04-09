import sys
import os
import dill
import numpy as np
import traceback

# Safe import for torch in case data contains tensors, but don't fail if missing
try:
    import torch
except ImportError:
    torch = None

# Add current directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_forward_operator import forward_operator
from verification_utils import recursive_check

# -------------------------------------------------------------------------
# HELPER INJECTION
# We need to inject the helper functions (C, CT, M_func, precompute_H_fft) 
# into the global namespace. Dill might reference them if it serializes 
# closures, or the function execution depends on them being present.
# -------------------------------------------------------------------------
import numpy.fft as fft

def C(M_arr, full_size, sensor_size):
    top = (full_size[0] - sensor_size[0]) // 2
    left = (full_size[1] - sensor_size[1]) // 2
    return M_arr[top:top+sensor_size[0], left:left+sensor_size[1]]

def CT(b, full_size, sensor_size):
    pad_top = (full_size[0] - sensor_size[0]) // 2
    pad_left = (full_size[1] - sensor_size[1]) // 2
    out = np.zeros(full_size, dtype=b.dtype)
    out[pad_top:pad_top+sensor_size[0], pad_left:pad_left+sensor_size[1]] = b
    return out

def M_func(vk, H_fft):
    return np.real(fft.fftshift(fft.ifft2(fft.fft2(fft.ifftshift(vk)) * H_fft)))

def precompute_H_fft(psf, full_size, sensor_size):
    return fft.fft2(fft.ifftshift(CT(psf, full_size, sensor_size)))

# Inject into global namespace
globals()['C'] = C
globals()['CT'] = CT
globals()['M_func'] = M_func
globals()['precompute_H_fft'] = precompute_H_fft

def run_test():
    # 1. Configuration
    data_paths = ['/data/yjh/DiffuserCam-Tutorial-master_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # Analyze Data Files
    outer_path = None
    inner_path = None
    
    for path in data_paths:
        if 'standard_data_forward_operator.pkl' in path:
            outer_path = path
        elif 'standard_data_parent_function_forward_operator' in path:
            inner_path = path

    if not outer_path:
        print("Error: standard_data_forward_operator.pkl not found.")
        sys.exit(1)

    try:
        # 2. Load Outer Data (Arguments for creating/running the operator)
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        
        # 3. Execute Target Function
        print(f"Executing forward_operator with args from {os.path.basename(outer_path)}...")
        actual_result = forward_operator(*outer_args, **outer_kwargs)

        # 4. Handle Factory vs Direct Execution logic
        # If inner_path exists, it means forward_operator returned a callable (Closure/Factory pattern)
        # If NOT, forward_operator returned the final result (Scenario A)
        
        if inner_path:
            # Scenario B: Factory Pattern
            if not callable(actual_result):
                print(f"Error: Expected callable return from forward_operator (due to existence of inner data file), but got {type(actual_result)}")
                sys.exit(1)
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
                
            inner_args = inner_data.get('args', [])
            inner_kwargs = inner_data.get('kwargs', {})
            expected_result = inner_data.get('output')
            
            print(f"Executing inner callable with args from {os.path.basename(inner_path)}...")
            final_actual = actual_result(*inner_args, **inner_kwargs)
            
        else:
            # Scenario A: Direct Execution
            final_actual = actual_result
            expected_result = outer_data.get('output')

        # 5. Verification
        passed, msg = recursive_check(expected_result, final_actual)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

    except Exception as e:
        print(f"Execution Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()