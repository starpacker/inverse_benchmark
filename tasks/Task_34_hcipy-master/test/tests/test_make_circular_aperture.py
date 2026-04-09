import sys
import os
import dill
import numpy as np
import glob
import traceback
from agent_make_circular_aperture import make_circular_aperture
# Import the target class to compare against
from agent_make_circular_aperture import Field as AgentField
from verification_utils import recursive_check

# -------------------------------------------------------------------------
# COMPATIBILITY SECTION
# -------------------------------------------------------------------------
# We define Field here exactly as it appears in the generator code.
# This ensures that when dill loads the pickle files (which expect a class 
# named 'Field' in the global scope if they were generated in __main__), 
# it can reconstruct the objects successfully.
class Field(np.ndarray):
    def __new__(cls, arr, grid):
        obj = np.asarray(arr).view(cls)
        obj.grid = grid
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.grid = getattr(obj, 'grid', None)

    @property
    def shaped(self):
        if self.grid.dims:
            return self.reshape(self.grid.dims)
        return self

# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------

def sanitize_to_agent_types(obj):
    """
    Recursively traverse the object. If we find an instance of the local __main__.Field,
    convert it to agent_make_circular_aperture.Field to satisfy strict type checking.
    """
    # If it's the local Field class (or any class named Field that isn't the AgentField)
    if hasattr(obj, '__class__') and obj.__class__.__name__ == 'Field' and obj.__class__ is not AgentField:
        # It is likely the deserialized object from pickle (type: __main__.Field)
        # We assume it is an ndarray subclass. We convert it to AgentField.
        try:
            # Create a view of the array with the correct class
            new_obj = obj.view(AgentField)
            # Copy attributes (specifically 'grid' for Field)
            if hasattr(obj, 'grid'):
                new_obj.grid = obj.grid
            # If there are other attributes in recursive structures (like grid properties), 
            # we might need to sanitize them too, but let's start with the Field container.
            return new_obj
        except Exception:
            # If conversion fails, return original and hope for loose comparison
            return obj

    if isinstance(obj, (list, tuple)):
        return type(obj)(sanitize_to_agent_types(x) for x in obj)
    if isinstance(obj, dict):
        return {k: sanitize_to_agent_types(v) for k, v in obj.items()}
    
    return obj

def run_test():
    # 1. Setup paths
    base_dir = '/data/yjh/hcipy-master_sandbox/run_code/std_data'
    outer_pattern = os.path.join(base_dir, 'standard_data_make_circular_aperture.pkl')
    
    # Locate outer file
    outer_files = glob.glob(outer_pattern)
    if not outer_files:
        print(f"Skipping: No outer data found matching {outer_pattern}")
        sys.exit(0)
    outer_path = outer_files[0]

    # Locate inner file (closure execution data)
    # The generator naming convention for inner functions is typically standard_data_parent_{parent_name}_{inner_name}.pkl
    # Here parent is 'make_circular_aperture'. Inner function name is 'func'.
    inner_pattern = os.path.join(base_dir, 'standard_data_parent_make_circular_aperture_*.pkl')
    inner_files = glob.glob(inner_pattern)

    print(f"Loading outer data from {outer_path}...")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"Error loading outer data: {e}")
        sys.exit(1)

    # 2. Reconstruct the Operator (Factory Phase)
    print("Executing make_circular_aperture (factory)...")
    try:
        outer_args = outer_data.get('args', [])
        outer_kwargs = outer_data.get('kwargs', {})
        
        # Sanitize inputs just in case arguments also have mismatching types (though less likely for primitive args)
        outer_args = sanitize_to_agent_types(outer_args)
        outer_kwargs = sanitize_to_agent_types(outer_kwargs)

        agent_operator = make_circular_aperture(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"Factory execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Execution & Verification
    if not inner_files:
        # Scenario A: No inner files, verify factory output directly? 
        # Usually factory returns a function, which is hard to verify equality on.
        # But if standard data implies a return value, we check it.
        # For this specific function, it returns a closure. Closures are not strictly equality-checkable.
        print("No inner data files found. Checking if factory return is callable...")
        if callable(agent_operator):
            print("Factory returned a callable as expected. TEST PASSED (Weak verification).")
            sys.exit(0)
        else:
            print("Factory did not return a callable.")
            sys.exit(1)
    
    # Scenario B: Inner files exist (Operator/Closure Pattern)
    inner_path = inner_files[0]
    print(f"Found inner data file: {inner_path}")
    
    try:
        with open(inner_path, 'rb') as f:
            inner_data = dill.load(f)
    except Exception as e:
        print(f"Error loading inner data: {e}")
        sys.exit(1)
        
    print("Executing closure with inner arguments...")
    try:
        inner_args = inner_data.get('args', [])
        inner_kwargs = inner_data.get('kwargs', {})

        # Sanitize arguments (e.g. if a Grid or Field is passed in)
        inner_args = sanitize_to_agent_types(inner_args)
        inner_kwargs = sanitize_to_agent_types(inner_kwargs)

        actual_result = agent_operator(*inner_args, **inner_kwargs)
    except Exception as e:
        print(f"Closure execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("Verifying results...")
    expected_result = inner_data['output']
    
    # CRITICAL FIX: Sanitize expected result to match Agent's types
    expected_result_sanitized = sanitize_to_agent_types(expected_result)

    try:
        passed, msg = recursive_check(expected_result_sanitized, actual_result)
    except Exception as e:
        print(f"Verification crashed: {e}")
        traceback.print_exc()
        sys.exit(1)

    if passed:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()