import sys
import os
import dill
import numpy as np
import traceback

# Add current directory to path to ensure imports work
sys.path.append(os.getcwd())

try:
    from agent_forward_operator import forward_operator
except ImportError:
    print("Error: Could not import forward_operator from agent_forward_operator.py")
    sys.exit(1)

try:
    from verification_utils import recursive_check
except ImportError:
    # Fallback if verification_utils is not immediately available, though prompt says it is.
    # We define a simple recursive check just in case, or fail if strictly required.
    # Assuming strict adherence to prompt, we assume it exists.
    print("Error: Could not import recursive_check from verification_utils")
    sys.exit(1)

def test_forward_operator():
    data_paths = ['/data/yjh/PtyLab-main_sandbox/run_code/std_data/standard_data_forward_operator.pkl']
    
    # 1. Identify the correct data file
    target_path = None
    for path in data_paths:
        if 'standard_data_forward_operator.pkl' in path:
            target_path = path
            break
            
    if not target_path or not os.path.exists(target_path):
        print(f"Error: Data file not found at {target_path}")
        sys.exit(1)
        
    # 2. Load the data
    try:
        with open(target_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        sys.exit(1)
        
    # 3. Extract inputs and expected output
    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_output = data.get('output', None)
    
    print(f"Loaded data for function: {data.get('func_name')}")
    
    # 4. Execute the function under test
    try:
        # Based on the provided code, forward_operator is a direct function, 
        # not a factory returning a closure. Scenario A applies.
        actual_result = forward_operator(*args, **kwargs)
    except Exception as e:
        print("Error during function execution:")
        traceback.print_exc()
        sys.exit(1)
        
    # 5. Verify results
    try:
        is_match, msg = recursive_check(expected_output, actual_result)
    except Exception as e:
        print(f"Error during verification: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    if is_match:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    test_forward_operator()