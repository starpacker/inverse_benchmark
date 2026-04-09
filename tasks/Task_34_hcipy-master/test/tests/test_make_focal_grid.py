import sys
import os
import dill
import numpy as np
import traceback
from agent_make_focal_grid import make_focal_grid
from verification_utils import recursive_check

# --- MOCK CLASSES FOR DILL COMPATIBILITY ---
# The pickle file contains objects (CartesianGrid) defined in the original generation script's __main__ scope.
# To successfully unpickle them without an AttributeError or ModuleNotFoundError, we must define 
# compatible stubs here in the test script's __main__ scope.

class Grid(object):
    def __init__(self, coords, weights=None):
        self.coords = coords
        self.weights = weights
        self._input_grid = None

class CartesianGrid(Grid):
    def __init__(self, coords, weights=None):
        super().__init__(coords, weights)
        self.is_regular = True
        self.dims = None
        self.delta = None

class Field(np.ndarray):
    def __new__(cls, arr, grid):
        obj = np.asarray(arr).view(cls)
        obj.grid = grid
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.grid = getattr(obj, 'grid', None)

# --- COMPARISON HELPERS ---

def to_pure_numpy(obj):
    """Convert Field objects or other array subclasses to pure numpy arrays."""
    if isinstance(obj, np.ndarray):
        return np.asarray(obj)
    if isinstance(obj, list):
        return [to_pure_numpy(x) for x in obj]
    return obj

def extract_grid_attributes(grid_obj):
    """
    Deconstructs a Grid/CartesianGrid object into a standard dictionary.
    
    This is CRITICAL to solving the 'Type mismatch' error.
    The 'expected' object is of type `__main__.CartesianGrid` (from pickle).
    The 'actual' object is of type `agent_make_focal_grid.CartesianGrid`.
    Direct comparison fails. Comparing their content (coords, weights, etc.) works.
    """
    if grid_obj is None:
        return None
    
    # grid_obj.coords is typically a list of numpy arrays
    coords = [to_pure_numpy(c) for c in grid_obj.coords]
    
    # grid_obj.weights might be None or an array
    weights = to_pure_numpy(grid_obj.weights) if grid_obj.weights is not None else None
    
    # CartesianGrid specific attributes
    dims = getattr(grid_obj, 'dims', None)
    delta = getattr(grid_obj, 'delta', None)
    
    return {
        'coords': coords,
        'weights': weights,
        'dims': dims,
        'delta': delta
    }

# --- MAIN TEST LOGIC ---

if __name__ == "__main__":
    # Hardcoded path based on instruction analysis
    data_paths = ['/data/yjh/hcipy-master_sandbox/run_code/std_data/standard_data_make_focal_grid.pkl']
    
    try:
        # 1. Validation of inputs
        if not data_paths:
            print("No data paths provided.")
            sys.exit(1)
            
        file_path = data_paths[0]
        if not os.path.exists(file_path):
            print(f"Data file not found: {file_path}")
            sys.exit(1)
            
        # 2. Load the Reference Data
        print(f"Loading data from {file_path}...")
        with open(file_path, 'rb') as f:
            # this will use the classes defined above to load the objects
            data = dill.load(f)
            
        args = data.get('args', [])
        kwargs = data.get('kwargs', {})
        expected_obj = data.get('output')
        
        print(f"Executing make_focal_grid with args: {len(args)} items, kwargs: {list(kwargs.keys())}")
        
        # 3. Execute the Function Under Test
        actual_obj = make_focal_grid(*args, **kwargs)
        
        # 4. Normalize Data for Verification
        # We convert both the deserialized 'expected' object and the 'actual' object
        # into dictionaries of pure numpy arrays/primitives to bypass class mismatch.
        print("Normalizing objects for robust comparison...")
        flat_expected = extract_grid_attributes(expected_obj)
        flat_actual = extract_grid_attributes(actual_obj)
        
        # 5. Verification
        passed, msg = recursive_check(flat_expected, flat_actual)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"Verification Failed: {msg}")
            sys.exit(1)

    except Exception as e:
        print(f"Execution failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)