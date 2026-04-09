import sys
import os
import dill
import numpy as np
import traceback
import warnings

# Handle optional torch import to prevent ModuleNotFoundError
try:
    import torch
except ImportError:
    torch = None

# Add current directory to sys.path so we can import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the target function
try:
    from agent__determine_aif_logic import _determine_aif_logic
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import target function. {e}")
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except ImportError:
    # Define a robust fallback if the utility is missing in the specific env
    def recursive_check(expected, actual):
        try:
            if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
                if len(expected) != len(actual):
                    return False, f"Length mismatch: {len(expected)} vs {len(actual)}"
                for i, (e, a) in enumerate(zip(expected, actual)):
                    valid, msg = recursive_check(e, a)
                    if not valid:
                        return False, f"Item {i}: {msg}"
                return True, "Matches"
            
            if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
                # Handle boolean arrays specifically (masks)
                if getattr(expected, 'dtype', None) == bool or getattr(actual, 'dtype', None) == bool:
                    if np.array_equal(expected, actual):
                        return True, "Boolean arrays match"
                    else:
                        return False, "Boolean array mismatch"
                
                # Handle numeric arrays with tolerance
                if np.allclose(expected, actual, equal_nan=True, atol=1e-4):
                    return True, "Arrays match"
                else:
                    diff = np.abs(np.array(expected) - np.array(actual))
                    return False, f"Array mismatch. Max diff: {np.max(diff)}"

            if expected == actual:
                return True, "Values match"
            
            return False, f"Value mismatch: {expected} != {actual}"
        except Exception as e:
            return False, f"Comparison error: {e}"

def run_test():
    data_paths = ['/data/yjh/CT-and-MR-Perfusion-Tool-main_sandbox/run_code/std_data/standard_data__determine_aif_logic.pkl']
    
    # Identify the data file. 
    # Based on the function name, we look for 'standard_data__determine_aif_logic.pkl'
    target_path = None
    for p in data_paths:
        if 'standard_data__determine_aif_logic.pkl' in p:
            target_path = p
            break
            
    if not target_path or not os.path.exists(target_path):
        print(f"Test skipped: Data file not found at {target_path}")
        sys.exit(0)

    try:
        print(f"Loading data from {target_path}...")
        with open(target_path, 'rb') as f:
            data = dill.load(f)
            
        args = data.get('args', [])
        kwargs = data.get('kwargs', {})
        expected_output = data.get('output', None)
        
        print("Executing _determine_aif_logic...")
        actual_output = _determine_aif_logic(*args, **kwargs)
        
        print("Verifying results...")
        passed, msg = recursive_check(expected_output, actual_output)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            # Debugging output shape info
            if hasattr(actual_output, '__len__'):
                print(f"Actual shape/len: {len(actual_output)}")
            if hasattr(expected_output, '__len__'):
                print(f"Expected shape/len: {len(expected_output)}")
            sys.exit(1)

    except Exception as e:
        print("TEST FAILED with Exception:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()