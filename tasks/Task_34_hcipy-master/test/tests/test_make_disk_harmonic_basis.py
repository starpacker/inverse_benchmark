import sys
import os
import dill
import numpy as np
import traceback

# Add current directory to path so we can import the agent
sys.path.append(os.path.dirname(__file__))

# Import the target function and associated classes
try:
    from agent_make_disk_harmonic_basis import make_disk_harmonic_basis, Field, ModeBasis
except ImportError:
    # If classes aren't exported, we might need to import * or access via module
    import agent_make_disk_harmonic_basis as agent_module
    make_disk_harmonic_basis = agent_module.make_disk_harmonic_basis
    Field = agent_module.Field
    ModeBasis = agent_module.ModeBasis

from verification_utils import recursive_check

# -------------------------------------------------------------------------
# Helper: Class Reconciliation
# -------------------------------------------------------------------------
def reconcile_types(obj):
    """
    Recursively convert instances of 'Field' or 'ModeBasis' from the pickle (which might be 
    seen as __main__.Field) to the actual classes imported from agent_make_disk_harmonic_basis.
    This resolves "Type mismatch: expected __main__.Field, got agent...Field".
    """
    if isinstance(obj, list):
        # ModeBasis is a subclass of list, check it first
        if type(obj).__name__ == 'ModeBasis':
            # Reconstruct ModeBasis using agent's class
            # ModeBasis usually has .grid attribute
            reconstructed_items = [reconcile_types(x) for x in obj]
            new_obj = ModeBasis(reconstructed_items)
            if hasattr(obj, 'grid'):
                new_obj.grid = obj.grid # Grid is likely a generic object or dict, leave as is
            return new_obj
        return [reconcile_types(x) for x in obj]
    
    if isinstance(obj, tuple):
        return tuple(reconcile_types(x) for x in obj)
    
    if isinstance(obj, dict):
        return {k: reconcile_types(v) for k, v in obj.items()}
    
    if isinstance(obj, np.ndarray):
        # Field is a subclass of ndarray
        if type(obj).__name__ == 'Field':
            # Create a new Field using the agent's class
            # Field(arr, grid)
            grid = getattr(obj, 'grid', None)
            return Field(obj, grid)
            
    return obj

# -------------------------------------------------------------------------
# Test Script
# -------------------------------------------------------------------------

def run_test():
    # 1. Define Data Paths
    data_paths = ['/data/yjh/hcipy-master_sandbox/run_code/std_data/standard_data_make_disk_harmonic_basis.pkl']
    
    # 2. Identify Outer and Inner Data
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        if 'parent_function' in p:
            inner_paths.append(p)
        elif 'make_disk_harmonic_basis.pkl' in p:
            outer_path = p

    if not outer_path:
        print("Error: standard_data_make_disk_harmonic_basis.pkl not found in paths.")
        sys.exit(1)

    # 3. Load Outer Data
    print(f"Loading outer data from {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', [])
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output_raw = outer_data.get('output', None)

    # Reconcile expected output types to match agent's classes
    expected_output = reconcile_types(expected_output_raw)

    # 4. Run the Agent Function (Scenario A logic, as this function returns a Basis (list subclass))
    # It does not return a closure/function, so we compare results directly.
    print(f"Running make_disk_harmonic_basis with args: {len(outer_args)} items, kwargs: {list(outer_kwargs.keys())}")
    
    try:
        actual_result = make_disk_harmonic_basis(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 5. Verification
    print("Verifying result against expected output...")
    
    # Debugging types if verification fails
    # print(f"Expected type: {type(expected_output)}")
    # print(f"Actual type: {type(actual_result)}")
    # if isinstance(expected_output, list) and len(expected_output) > 0:
    #      print(f"Expected[0] type: {type(expected_output[0])}")
    #      print(f"Actual[0] type: {type(actual_result[0])}")

    passed, msg = recursive_check(expected_output, actual_result)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()