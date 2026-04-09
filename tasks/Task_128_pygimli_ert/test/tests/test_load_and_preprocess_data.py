import sys
import os
import dill
import numpy as np
import traceback

# Add the working directory to path
sys.path.insert(0, '/data/yjh/pygimli_ert_sandbox_sandbox/run_code')
sys.path.insert(0, '/data/yjh/pygimli_ert_sandbox_sandbox')

from agent_load_and_preprocess_data import load_and_preprocess_data
from verification_utils import recursive_check


def main():
    data_paths = ['/data/yjh/pygimli_ert_sandbox_sandbox/run_code/std_data/standard_data_load_and_preprocess_data.pkl']

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    # Try to load the outer data file
    outer_data = None
    if outer_path and os.path.exists(outer_path):
        file_size = os.path.getsize(outer_path)
        if file_size > 0:
            try:
                with open(outer_path, 'rb') as f:
                    outer_data = dill.load(f)
                print(f"Successfully loaded outer data from {outer_path} (size={file_size})")
            except Exception as e:
                print(f"Warning: Could not load outer data: {e}")
                outer_data = None
        else:
            print(f"Warning: Outer data file is empty (size={file_size})")

    # Determine args/kwargs for calling the function
    if outer_data is not None and isinstance(outer_data, dict):
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)
    else:
        # File is corrupted/empty - use known default input from the reference code
        # The function takes a single argument: results_dir
        print("Using fallback arguments since pkl file could not be loaded.")
        test_results_dir = '/data/yjh/pygimli_ert_sandbox_sandbox/run_code/results_ert_benchmark'
        outer_args = (test_results_dir,)
        outer_kwargs = {}
        expected_output = None

    # Phase 1: Call the function
    print("\n--- Phase 1: Calling load_and_preprocess_data ---")
    try:
        result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print("Function executed successfully.")
    except Exception as e:
        print(f"FAIL: Function execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Check for inner paths (Scenario B)
    if inner_paths:
        print("\n--- Phase 2: Scenario B - Factory/Closure Pattern ---")
        for ip in inner_paths:
            if not os.path.exists(ip):
                print(f"Warning: Inner path does not exist: {ip}")
                continue
            try:
                file_size = os.path.getsize(ip)
                if file_size == 0:
                    print(f"Warning: Inner data file is empty: {ip}")
                    continue
                with open(ip, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output', None)

                if not callable(result):
                    print(f"FAIL: Expected callable from Phase 1, got {type(result)}")
                    sys.exit(1)

                actual_result = result(*inner_args, **inner_kwargs)
                passed, msg = recursive_check(inner_expected, actual_result)
                if not passed:
                    print(f"FAIL: {msg}")
                    sys.exit(1)
                else:
                    print("Inner test passed.")
            except Exception as e:
                print(f"FAIL: Inner execution error: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function
        print("\n--- Phase 2: Scenario A - Simple Function ---")
        if expected_output is not None:
            try:
                passed, msg = recursive_check(expected_output, result)
                if not passed:
                    print(f"FAIL: {msg}")
                    sys.exit(1)
                else:
                    print("Comparison with expected output passed.")
            except Exception as e:
                print(f"FAIL: Comparison error: {e}")
                traceback.print_exc()
                sys.exit(1)
        else:
            # Cannot compare with expected output - validate structure and properties
            print("No expected output available. Validating output structure and properties...")
            try:
                assert isinstance(result, dict), f"Expected dict, got {type(result)}"

                required_keys = ['mesh', 'scheme', 'geom', 'rhomap', 'gt_res_np', 'results_dir']
                for key in required_keys:
                    assert key in result, f"Missing key: {key}"
                print(f"  All required keys present: {required_keys}")

                # Validate rhomap
                expected_rhomap = [
                    [0, 100.0], [1, 100.0], [2, 50.0], [3, 200.0],
                    [4, 10.0], [5, 500.0], [6, 5.0],
                ]
                assert result['rhomap'] == expected_rhomap, \
                    f"rhomap mismatch: {result['rhomap']} != {expected_rhomap}"
                print("  rhomap matches expected values.")

                # Validate gt_res_np
                gt = result['gt_res_np']
                assert isinstance(gt, np.ndarray), f"gt_res_np should be ndarray, got {type(gt)}"
                assert gt.min() == 5.0, f"gt_res_np min should be 5.0, got {gt.min()}"
                assert gt.max() == 500.0, f"gt_res_np max should be 500.0, got {gt.max()}"
                print(f"  gt_res_np: shape={gt.shape}, min={gt.min()}, max={gt.max()}")

                # Validate results_dir
                assert result['results_dir'] == outer_args[0], \
                    f"results_dir mismatch: {result['results_dir']} != {outer_args[0]}"
                print(f"  results_dir matches: {result['results_dir']}")

                # Validate scheme
                scheme = result['scheme']
                assert scheme.sensorCount() == 41, \
                    f"Expected 41 sensors, got {scheme.sensorCount()}"
                print(f"  scheme: {scheme.sensorCount()} sensors, {scheme.size()} measurements")

                # Validate mesh
                mesh = result['mesh']
                assert mesh.cellCount() > 0, "Mesh should have cells"
                assert mesh.nodeCount() > 0, "Mesh should have nodes"
                print(f"  mesh: {mesh.cellCount()} cells, {mesh.nodeCount()} nodes")

                # Validate gt_res_np length matches mesh cells
                assert len(gt) == mesh.cellCount(), \
                    f"gt_res_np length ({len(gt)}) should match mesh cell count ({mesh.cellCount()})"
                print(f"  gt_res_np length matches mesh cell count: {len(gt)}")

                print("\nAll structural validations passed.")

            except AssertionError as e:
                print(f"FAIL: Validation error: {e}")
                traceback.print_exc()
                sys.exit(1)
            except Exception as e:
                print(f"FAIL: Unexpected error during validation: {e}")
                traceback.print_exc()
                sys.exit(1)

    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()