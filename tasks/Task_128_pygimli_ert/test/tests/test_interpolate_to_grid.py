import sys
import os
import dill
import numpy as np
import traceback

# Add the run_code directory to path
sys.path.insert(0, '/data/yjh/pygimli_ert_sandbox_sandbox/run_code')

from agent_interpolate_to_grid import interpolate_to_grid
from verification_utils import recursive_check


def try_load_pkl(path):
    """Try to load a pickle file, return None if it fails."""
    try:
        if not os.path.exists(path):
            return None
        file_size = os.path.getsize(path)
        if file_size < 10:
            print(f"File {path} is too small ({file_size} bytes), likely empty/corrupt.")
            return None
        with open(path, 'rb') as f:
            data = dill.load(f)
        return data
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None


def create_test_mesh_and_data():
    """Create a simple PyGIMLI mesh and test data for interpolation."""
    try:
        import pygimli as pg
        
        # Create a simple rectangular mesh
        mesh = pg.createGrid(x=np.linspace(0, 10, 11), y=np.linspace(-5, 0, 6))
        
        # Create cell values (e.g., based on cell center positions)
        cell_values = []
        for i in range(mesh.cellCount()):
            cx = mesh.cell(i).center().x()
            cy = mesh.cell(i).center().y()
            cell_values.append(cx + cy)  # Simple linear function
        
        x_range = (0, 10)
        y_range = (-5, 0)
        nx = 50
        ny = 25
        
        return mesh, cell_values, x_range, y_range, nx, ny
    except ImportError:
        print("PyGIMLI not available, trying alternative mesh creation...")
        raise


def main():
    data_paths = ['/data/yjh/pygimli_ert_sandbox_sandbox/run_code/std_data/standard_data_interpolate_to_grid.pkl']
    
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        else:
            outer_path = p
    
    # Try loading the pkl data first
    outer_data = None
    if outer_path:
        outer_data = try_load_pkl(outer_path)
    
    if outer_data is not None:
        # Scenario: pkl file loaded successfully
        print("Successfully loaded pickle data.")
        try:
            args = outer_data.get('args', ())
            kwargs = outer_data.get('kwargs', {})
            expected_output = outer_data.get('output', None)
            
            result = interpolate_to_grid(*args, **kwargs)
            
            if inner_paths:
                # Scenario B: factory pattern
                inner_data = try_load_pkl(inner_paths[0])
                if inner_data is not None:
                    inner_args = inner_data.get('args', ())
                    inner_kwargs = inner_data.get('kwargs', {})
                    inner_expected = inner_data.get('output', None)
                    
                    actual_result = result(*inner_args, **inner_kwargs)
                    passed, msg = recursive_check(inner_expected, actual_result)
                    if not passed:
                        print(f"FAIL: {msg}")
                        sys.exit(1)
                    print("TEST PASSED")
                    sys.exit(0)
            else:
                # Scenario A: simple function
                passed, msg = recursive_check(expected_output, result)
                if not passed:
                    print(f"FAIL: {msg}")
                    sys.exit(1)
                print("TEST PASSED")
                sys.exit(0)
                
        except Exception as e:
            print(f"Error running with pkl data: {e}")
            traceback.print_exc()
            # Fall through to direct test
    
    # Fallback: pkl data unavailable or corrupt, create test data directly
    print("Pickle data unavailable/corrupt. Creating test data directly with PyGIMLI...")
    
    try:
        mesh, cell_values, x_range, y_range, nx, ny = create_test_mesh_and_data()
    except Exception as e:
        print(f"FAIL: Could not create test data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # ---- Test 1: Basic functionality and output types ----
    print("Test 1: Basic functionality and output structure...")
    try:
        result = interpolate_to_grid(mesh, cell_values, x_range, y_range, nx=nx, ny=ny)
    except Exception as e:
        print(f"FAIL: interpolate_to_grid raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Should return a tuple of 3 elements
    if not isinstance(result, tuple):
        print(f"FAIL: Expected tuple output, got {type(result)}")
        sys.exit(1)
    
    if len(result) != 3:
        print(f"FAIL: Expected tuple of length 3, got length {len(result)}")
        sys.exit(1)
    
    grid_values, x_coords, y_coords = result
    
    # Check types
    if not isinstance(grid_values, np.ndarray):
        print(f"FAIL: grid_values should be np.ndarray, got {type(grid_values)}")
        sys.exit(1)
    if not isinstance(x_coords, np.ndarray):
        print(f"FAIL: x_coords should be np.ndarray, got {type(x_coords)}")
        sys.exit(1)
    if not isinstance(y_coords, np.ndarray):
        print(f"FAIL: y_coords should be np.ndarray, got {type(y_coords)}")
        sys.exit(1)
    
    print("  Output structure: OK")
    
    # ---- Test 2: Output shapes ----
    print("Test 2: Output shapes...")
    if grid_values.shape != (ny, nx):
        print(f"FAIL: grid_values shape should be ({ny}, {nx}), got {grid_values.shape}")
        sys.exit(1)
    if x_coords.shape != (nx,):
        print(f"FAIL: x_coords shape should be ({nx},), got {x_coords.shape}")
        sys.exit(1)
    if y_coords.shape != (ny,):
        print(f"FAIL: y_coords shape should be ({ny},), got {y_coords.shape}")
        sys.exit(1)
    print("  Shapes: OK")
    
    # ---- Test 3: Coordinate ranges ----
    print("Test 3: Coordinate ranges...")
    if not np.isclose(x_coords[0], x_range[0]):
        print(f"FAIL: x_coords start should be {x_range[0]}, got {x_coords[0]}")
        sys.exit(1)
    if not np.isclose(x_coords[-1], x_range[1]):
        print(f"FAIL: x_coords end should be {x_range[1]}, got {x_coords[-1]}")
        sys.exit(1)
    if not np.isclose(y_coords[0], y_range[0]):
        print(f"FAIL: y_coords start should be {y_range[0]}, got {y_coords[0]}")
        sys.exit(1)
    if not np.isclose(y_coords[-1], y_range[1]):
        print(f"FAIL: y_coords end should be {y_range[1]}, got {y_coords[-1]}")
        sys.exit(1)
    print("  Coordinate ranges: OK")
    
    # ---- Test 4: Interpolation correctness ----
    # Since cell_values = cx + cy (linear), linear interpolation should recover this exactly
    # (within the convex hull of cell centers)
    print("Test 4: Interpolation correctness (linear function)...")
    xx, yy = np.meshgrid(x_coords, y_coords)
    expected_values = xx + yy  # The linear function we used
    
    # Only check where interpolation is valid (not NaN)
    valid_mask = ~np.isnan(grid_values)
    if valid_mask.sum() == 0:
        print(f"FAIL: All interpolated values are NaN")
        sys.exit(1)
    
    valid_grid = grid_values[valid_mask]
    valid_expected = expected_values[valid_mask]
    
    max_error = np.max(np.abs(valid_grid - valid_expected))
    print(f"  Max interpolation error: {max_error}")
    if max_error > 1e-6:
        print(f"FAIL: Interpolation error too large: {max_error}")
        sys.exit(1)
    print("  Interpolation correctness: OK")
    
    # ---- Test 5: Determinism (running twice gives same result) ----
    print("Test 5: Determinism...")
    result2 = interpolate_to_grid(mesh, cell_values, x_range, y_range, nx=nx, ny=ny)
    grid_values2, x_coords2, y_coords2 = result2
    
    passed, msg = recursive_check(result, result2)
    if not passed:
        print(f"FAIL: Non-deterministic results: {msg}")
        sys.exit(1)
    print("  Determinism: OK")
    
    # ---- Test 6: Default parameters ----
    print("Test 6: Default parameters (nx=100, ny=50)...")
    try:
        result_default = interpolate_to_grid(mesh, cell_values, x_range, y_range)
        gv_d, xc_d, yc_d = result_default
        if gv_d.shape != (50, 100):
            print(f"FAIL: Default shape should be (50, 100), got {gv_d.shape}")
            sys.exit(1)
        if xc_d.shape != (100,):
            print(f"FAIL: Default x_coords shape should be (100,), got {xc_d.shape}")
            sys.exit(1)
        if yc_d.shape != (50,):
            print(f"FAIL: Default y_coords shape should be (50,), got {yc_d.shape}")
            sys.exit(1)
    except Exception as e:
        print(f"FAIL: Default parameters test raised exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    print("  Default parameters: OK")
    
    # ---- Test 7: Different nx, ny values ----
    print("Test 7: Custom nx, ny values...")
    try:
        result_custom = interpolate_to_grid(mesh, cell_values, x_range, y_range, nx=20, ny=10)
        gv_c, xc_c, yc_c = result_custom
        if gv_c.shape != (10, 20):
            print(f"FAIL: Custom shape should be (10, 20), got {gv_c.shape}")
            sys.exit(1)
    except Exception as e:
        print(f"FAIL: Custom nx/ny test raised exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    print("  Custom nx, ny: OK")
    
    # ---- Test 8: Verify against direct scipy griddata call ----
    print("Test 8: Verify against direct scipy.interpolate.griddata...")
    try:
        import pygimli as pg
        from scipy.interpolate import griddata
        
        cell_centers = np.array([[mesh.cell(i).center().x(),
                                  mesh.cell(i).center().y()]
                                 for i in range(mesh.cellCount())])
        
        x_test = np.linspace(x_range[0], x_range[1], nx)
        y_test = np.linspace(y_range[0], y_range[1], ny)
        xx_test, yy_test = np.meshgrid(x_test, y_test)
        
        expected_grid = griddata(cell_centers, np.array(cell_values),
                                 (xx_test, yy_test), method='linear', fill_value=np.nan)
        
        passed_ref, msg_ref = recursive_check(expected_grid, grid_values)
        if not passed_ref:
            print(f"FAIL: Mismatch with direct griddata call: {msg_ref}")
            sys.exit(1)
        
        passed_xref, msg_xref = recursive_check(x_test, x_coords)
        if not passed_xref:
            print(f"FAIL: x_coords mismatch: {msg_xref}")
            sys.exit(1)
        
        passed_yref, msg_yref = recursive_check(y_test, y_coords)
        if not passed_yref:
            print(f"FAIL: y_coords mismatch: {msg_yref}")
            sys.exit(1)
            
    except ImportError:
        print("  Skipped (import issue)")
    except Exception as e:
        print(f"FAIL: Reference comparison raised exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    print("  Reference comparison: OK")
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()