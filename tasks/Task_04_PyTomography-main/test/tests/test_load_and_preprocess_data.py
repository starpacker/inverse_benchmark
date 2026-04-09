import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function and the custom classes associated with its output
try:
    from agent_load_and_preprocess_data import load_and_preprocess_data, SPECTObjectMeta, SPECTProjMeta
except ImportError:
    # Fallback if running in an environment where the module needs path adjustment
    sys.path.append(os.getcwd())
    from agent_load_and_preprocess_data import load_and_preprocess_data, SPECTObjectMeta, SPECTProjMeta

from verification_utils import recursive_check

# Data paths provided in the prompt
data_paths = ['/data/yjh/PyTomography-main_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']

def sanitize_for_comparison(obj):
    """
    Recursively converts custom objects (SPECTObjectMeta, SPECTProjMeta) 
    into dictionaries to allow attribute-based comparison by recursive_check.
    This fixes the issue where custom classes without __eq__ fail equality checks.
    """
    # Handle the specific classes known to cause issues
    if isinstance(obj, (SPECTObjectMeta, SPECTProjMeta)):
        return sanitize_for_comparison(obj.__dict__)
    
    # Generic handler for other custom objects that might be nested
    # We check if it has a __dict__ and is not a standard primitive or tensor
    if hasattr(obj, '__dict__') and not isinstance(obj, (torch.Tensor, np.ndarray, int, float, str, bool, type(None))):
         try:
             return sanitize_for_comparison(obj.__dict__)
         except:
             pass

    # Recursive handling for standard containers
    if isinstance(obj, dict):
        return {k: sanitize_for_comparison(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_comparison(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_for_comparison(x) for x in obj)
        
    return obj

def run_test():
    outer_path = None
    
    # Locate the standard data file
    for p in data_paths:
        if 'standard_data_load_and_preprocess_data.pkl' in p:
            outer_path = p
            break

    if not outer_path:
        print("Error: No standard data file found.")
        sys.exit(1)

    try:
        print(f"Loading data from {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)

        args = outer_data.get('args', [])
        kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)

        print("Executing load_and_preprocess_data...")
        
        # Execute the function
        # Note: The function returns a tuple of (data, meta, meta, float), it is not a factory.
        actual_output = load_and_preprocess_data(*args, **kwargs)

        print("Execution complete. Verifying results...")

        # FIX: The failure "Value mismatch at output[1]" occurs because SPECTObjectMeta
        # does not implement __eq__. We convert these objects to dicts of their attributes
        # before passing them to the verification utility.
        clean_expected = sanitize_for_comparison(expected_output)
        clean_actual = sanitize_for_comparison(actual_output)

        passed, msg = recursive_check(clean_expected, clean_actual)

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