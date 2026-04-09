import sys
import os
import dill
import numpy as np
import scipy.sparse as sp
import traceback

# Add necessary paths or imports if required by the environment
# Assuming agent_get_source_vector is in the python path or current directory
try:
    from agent_get_source_vector import get_source_vector
except ImportError:
    # Fallback: if the file is in the same dir but not in path
    sys.path.append(os.path.dirname(__file__))
    from agent_get_source_vector import get_source_vector

try:
    from verification_utils import recursive_check
except ImportError:
    # minimal fallback if verification_utils is missing in this specific context
    def recursive_check(expected, actual):
        if isinstance(expected, np.ndarray):
            if not np.allclose(expected, actual):
                return False, "Numpy arrays differ"
            return True, ""
        if expected != actual:
            return False, f"Expected {expected} != Actual {actual}"
        return True, ""

def custom_verification(expected, actual):
    """
    Custom verification logic to handle Scipy Sparse Matrices which cause
    ambiguity errors in standard recursive checks or equality operators.
    """
    # Check if both are sparse matrices
    if sp.issparse(expected) and sp.issparse(actual):
        # Check shape
        if expected.shape != actual.shape:
            return False, f"Shape mismatch: Expected {expected.shape}, Got {actual.shape}"
        
        # Check difference
        # Note: (A != B) returns a sparse boolean matrix, not a bool, which causes the crash.
        # We calculate the difference matrix and check if its values are close to zero.
        diff = (expected - actual)
        diff.eliminate_zeros() # Explicitly remove explicit zeros
        
        if diff.nnz == 0:
            return True, "Sparse matrices match exactly."
        
        # Check magnitude of differences (floating point tolerance)
        max_diff = np.max(np.abs(diff.data))
        if max_diff < 1e-5:
            return True, f"Sparse matrices match within tolerance (max diff: {max_diff})."
        else:
            return False, f"Sparse matrices content mismatch. Max diff: {max_diff}"

    # Check if one is sparse and the other is not
    if sp.issparse(expected) != sp.issparse(actual):
        return False, f"Type mismatch: Expected sparse={sp.issparse(expected)}, Actual sparse={sp.issparse(actual)}"

    # Fallback to standard recursive check for other types
    return recursive_check(expected, actual)

def run_test():
    # 1. Setup Data Paths
    data_paths = ['/data/yjh/nirfaster-FF-main_2_sandbox/run_code/std_data/standard_data_get_source_vector.pkl']
    
    if not data_paths:
        print("TEST FAILED: No data paths provided.")
        sys.exit(1)

    # We expect a simple function scenario based on the target function analysis
    outer_path = None
    for path in data_paths:
        if 'standard_data_get_source_vector.pkl' in path:
            outer_path = path
            break
            
    if not outer_path:
        print("TEST FAILED: Could not find standard_data_get_source_vector.pkl")
        sys.exit(1)

    # 2. Load Data
    print(f"Loading test data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"TEST FAILED: Failed to load pickle data. Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Extract Inputs
    args = data.get('args', [])
    kwargs = data.get('kwargs', {})
    expected_output = data.get('output', None)
    
    # 4. Execute Function
    print("Executing get_source_vector...")
    try:
        actual_output = get_source_vector(*args, **kwargs)
    except Exception as e:
        print(f"TEST FAILED: Execution raised exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 5. Verify Results
    print("Verifying results...")
    try:
        is_match, failure_msg = custom_verification(expected_output, actual_output)
        
        if is_match:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: Verification failed. {failure_msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"TEST FAILED: Verification utility crashed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_test()