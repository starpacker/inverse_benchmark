import sys
import os
import dill
import numpy as np
import scipy.sparse as sp
import traceback
import copy
import scipy.spatial as spatial

# Add the directory containing the agent code to the path
sys.path.append('/data/yjh/nirfaster-FF-main_2_sandbox/run_code')

from agent_load_and_preprocess_data import load_and_preprocess_data
import agent_load_and_preprocess_data

# -----------------------------------------------------------------------------
# DEFINITIONS FOR UNPICKLING CONTEXT
# -----------------------------------------------------------------------------
# We define these here because 'dill' might expect them to exist in __main__
# if the data was generated in a script running as __main__.

def boundary_attenuation(n_incidence, n_transmission=1.0):
    """Calculate the boundary attenuation factor A using Fresnel's law (Robin BC)."""
    n = n_incidence / n_transmission
    R0 = ((n - 1.) ** 2) / ((n + 1.) ** 2)
    theta_c = np.arcsin(1.0 / n)
    cos_theta_c = np.cos(theta_c)
    A = (2.0 / (1.0 - R0) - 1.0 + np.abs(cos_theta_c) ** 3) / (1.0 - np.abs(cos_theta_c) ** 2)
    return A

class StndMesh:
    def __init__(self):
        self.nodes = None
        self.elements = None
        self.bndvtx = None
        self.mua = None
        self.kappa = None
        self.ri = None
        self.mus = None
        self.ksi = None 
        self.c = None 
        self.source = {}
        self.meas = {}
        self.link = None
        self.dimension = 2
        self.vol = {}

    def copy_from(self, other):
        self.nodes = copy.deepcopy(other.nodes)
        self.elements = copy.deepcopy(other.elements)
        self.bndvtx = copy.deepcopy(other.bndvtx)
        self.mua = copy.deepcopy(other.mua)
        self.kappa = copy.deepcopy(other.kappa)
        self.ri = copy.deepcopy(other.ri)
        self.mus = copy.deepcopy(other.mus)
        self.ksi = copy.deepcopy(other.ksi)
        self.c = copy.deepcopy(other.c)
        self.source = copy.deepcopy(other.source)
        self.meas = copy.deepcopy(other.meas)
        self.link = copy.deepcopy(other.link)
        self.dimension = other.dimension
        self.vol = copy.deepcopy(other.vol)

# -----------------------------------------------------------------------------
# ROBUST VERIFICATION UTILS
# -----------------------------------------------------------------------------

def robust_compare(expected, actual, path=""):
    """
    Compares two objects recursively, being lenient about class namespaces 
    (e.g. __main__.StndMesh vs agent_module.StndMesh) and using 
    numpy-specific checks for arrays.
    """
    # 1. Handle None
    if expected is None:
        if actual is not None:
            return False, f"{path}: Expected None, got {type(actual)}"
        return True, ""

    # 2. Handle Numpy Arrays/Sparse Matrices
    if sp.issparse(expected):
        if not sp.issparse(actual):
            return False, f"{path}: Expected sparse matrix, got {type(actual)}"
        diff = (expected - actual).sum()
        if np.abs(diff) > 1e-5:
            return False, f"{path}: Sparse matrices differ. Sum diff: {diff}"
        return True, ""
        
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray):
            return False, f"{path}: Expected numpy array, got {type(actual)}"
        if expected.shape != actual.shape:
            return False, f"{path}: Shape mismatch {expected.shape} vs {actual.shape}"
        
        # String arrays or object arrays
        if expected.dtype == np.object_ or expected.dtype.type is np.str_:
            if not np.array_equal(expected, actual):
                return False, f"{path}: Object/String array content mismatch"
            return True, ""
            
        # Numeric arrays
        try:
            if not np.allclose(expected, actual, rtol=1e-4, atol=1e-4, equal_nan=True):
                # Calculate max diff for error message
                diff = np.abs(expected - actual)
                max_diff = np.max(diff)
                return False, f"{path}: Numerical mismatch. Max diff: {max_diff}"
        except Exception as e:
            return False, f"{path}: Array comparison failed: {e}"
        return True, ""

    # 3. Handle Lists/Tuples
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)):
            return False, f"{path}: Expected list/tuple, got {type(actual)}"
        if len(expected) != len(actual):
            return False, f"{path}: Length mismatch {len(expected)} vs {len(actual)}"
        for i, (e_item, a_item) in enumerate(zip(expected, actual)):
            ok, msg = robust_compare(e_item, a_item, path=f"{path}[{i}]")
            if not ok:
                return False, msg
        return True, ""

    # 4. Handle Dictionaries
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False, f"{path}: Expected dict, got {type(actual)}"
        if set(expected.keys()) != set(actual.keys()):
            return False, f"{path}: Keys mismatch. Expected {list(expected.keys())}, got {list(actual.keys())}"
        for k in expected:
            ok, msg = robust_compare(expected[k], actual[k], path=f"{path}['{k}']")
            if not ok:
                return False, msg
        return True, ""

    # 5. Handle Custom Objects (The main fix for Namespace issues)
    # Check if they have __dict__ and matching class names
    if hasattr(expected, '__dict__'):
        expected_name = expected.__class__.__name__
        actual_name = actual.__class__.__name__
        
        # If class names match (ignoring module), compare attributes
        if expected_name == actual_name:
            if not hasattr(actual, '__dict__'):
                return False, f"{path}: Expected object with __dict__, actual has none"
            
            # Compare attributes
            e_vars = vars(expected)
            a_vars = vars(actual)
            
            # Allow actual to have *more* attributes (e.g. if code evolved), 
            # but it must contain all expected attributes
            for k, v in e_vars.items():
                if k not in a_vars:
                    return False, f"{path}: Attribute '{k}' missing in actual object"
                ok, msg = robust_compare(v, a_vars[k], path=f"{path}.{k}")
                if not ok:
                    return False, msg
            return True, ""
        else:
            return False, f"{path}: Class name mismatch {expected_name} vs {actual_name}"

    # 6. Fallback: standard equality
    if expected != actual:
        return False, f"{path}: Value mismatch {expected} != {actual}"

    return True, ""


# -----------------------------------------------------------------------------
# MAIN TEST SCRIPT
# -----------------------------------------------------------------------------

def main():
    data_paths = ['/data/yjh/nirfaster-FF-main_2_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']
    
    # 1. Identify Data File
    # We are looking for the 'outer' data file for the function call.
    outer_path = None
    inner_path = None
    
    for p in data_paths:
        if 'standard_data_load_and_preprocess_data.pkl' in p:
            outer_path = p
        elif 'standard_data_parent_function_load_and_preprocess_data' in p:
            inner_path = p # Scenario B: Inner data
            
    if not outer_path:
        print("Error: No standard_data_load_and_preprocess_data.pkl found.")
        sys.exit(1)

    # 2. Load Data
    try:
        print(f"Loading data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            data = dill.load(f)
    except Exception as e:
        print(f"Failed to load pickle data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Execution Logic
    try:
        args = data.get('args', [])
        kwargs = data.get('kwargs', {})
        expected_output = data.get('output', None)
        
        print("Executing load_and_preprocess_data with loaded arguments...")
        actual_output = load_and_preprocess_data(*args, **kwargs)
        
        # 4. Scenario Check
        # If there was an "inner_path", the function would have returned a closure/operator 
        # that we then need to execute. However, looking at the code for 
        # load_and_preprocess_data, it returns (mesh, mesh_anomaly, grid_info) 
        # directly, not a closure. So this is likely Scenario A (Simple Function).
        
        # We verify the result immediately.
        
        print("Verifying results...")
        passed, msg = robust_compare(expected_output, actual_output, path="output")
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)

    except Exception as e:
        print(f"Execution or Verification failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()