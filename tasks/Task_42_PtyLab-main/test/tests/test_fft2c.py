import sys
import os
import dill
import numpy as np
import traceback

# Add the directory containing the agent code to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the target function
from agent_fft2c import fft2c
from verification_utils import recursive_check

def test_fft2c():
    """
    Unit test for fft2c using captured standard data.
    """
    # 1. Define Data Paths
    # The prompt provides a list of paths. We need to identify the relevant files.
    # Based on the decorator logic:
    # - Standard data: standard_data_fft2c.pkl
    # - Parent/Inner data (if it returns a closure): standard_data_parent_function_fft2c_*.pkl
    
    data_dir = '/data/yjh/PtyLab-main_sandbox/run_code/std_data'
    outer_data_path = os.path.join(data_dir, 'standard_data_fft2c.pkl')
    
    # Check if outer data exists
    if not os.path.exists(outer_data_path):
        print(f"Error: Data file not found at {outer_data_path}")
        sys.exit(1)

    # 2. Load Outer Data
    try:
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer data: {e}")
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_result = outer_data.get('output')

    print(f"Loaded data for function: {outer_data.get('func_name')}")

    # 3. Execute the function
    # fft2c seems to be a direct mathematical function based on the provided code,
    # not a closure factory. It returns the result directly.
    try:
        actual_result = fft2c(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Error executing fft2c: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Scenario Handling
    # In some advanced scenarios (Scenario B), the function might return a callable (closure).
    # The decorator logic handles this by creating a secondary file.
    # However, looking at the source of fft2c provided in the prompt context:
    # def fft2c(x): return scipy.fft.fftshift(scipy.fft.fft2(scipy.fft.ifftshift(x)))
    # It returns a result directly (likely a numpy array), not a function.
    
    # We check if there are any inner data files just in case the captured behavior implies a closure,
    # but strictly following the provided source, it's Scenario A (Direct Execution).
    
    # If the result is callable, we look for inner data files.
    if callable(actual_result) and not isinstance(actual_result, (np.ndarray, list, tuple, dict)):
        print("Detected callable output (Scenario B - Closure). Looking for inner execution data...")
        
        # Find inner data file
        inner_files = [f for f in os.listdir(data_dir) if f.startswith('standard_data_parent_fft2c_')]
        if not inner_files:
            print("Error: Function returned a callable, but no inner execution data found.")
            sys.exit(1)
            
        inner_path = os.path.join(data_dir, inner_files[0]) # Take the first one found
        print(f"Loading inner data from: {inner_path}")
        
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
        except Exception as e:
            print(f"Error loading inner data: {e}")
            sys.exit(1)
            
        inner_args = inner_data.get('args', [])
        inner_kwargs = inner_data.get('kwargs', {})
        expected_result = inner_data.get('output')
        
        try:
            actual_result = actual_result(*inner_args, **inner_kwargs)
        except Exception as e:
            print(f"Error executing inner closure: {e}")
            traceback.print_exc()
            sys.exit(1)

    # 5. Verification
    print("Verifying results...")
    is_correct, fail_msg = recursive_check(expected_result, actual_result)

    if is_correct:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {fail_msg}")
        sys.exit(1)

if __name__ == "__main__":
    test_fft2c()